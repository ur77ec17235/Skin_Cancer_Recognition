"""
Advanced Training Optimizations for Skin Cancer Recognition
Implements: AMP, Gradient Checkpointing, Early Stopping, Model Pruning, etc.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.utils.checkpoint as checkpoint
import torch.nn.utils.prune as prune
from pathlib import Path
from typing import Dict, Tuple
import gc


# ============================================
# 1. MIXED PRECISION (AMP)
# ============================================

class AMPTrainer:
    """
    Automatic Mixed Precision Training
    - Uses float16 for forward/backward
    - Maintains float32 for weights
    - Reduces VRAM by ~50%, increases speed by 1.5-2x
    """
    def __init__(self, model, use_amp: bool = True):
        self.model = model
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
    
    def forward_backward(self, images, labels, criterion, optimizer):
        """Forward pass with AMP"""
        optimizer.zero_grad(set_to_none=True)
        
        if self.use_amp:
            with autocast(device_type='cuda'):
                outputs = self.model(images)
                loss = criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
        
        return loss.item(), outputs


# ============================================
# 2. GRADIENT CHECKPOINTING
# ============================================

class GradientCheckpointingWrapper(nn.Module):
    """
    Wraps model for gradient checkpointing
    - Doesn't cache activations during forward pass
    - Recomputes them during backward
    - Saves 30-40% memory at cost of slightly more compute
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    
    def forward(self, x, use_checkpoint=True):
        if use_checkpoint and self.training:
            return checkpoint.checkpoint(
                self._forward_impl,
                x,
                use_reentrant=False
            )
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x):
        return self.backbone(x)


# ============================================
# 3. EARLY STOPPING
# ============================================

class EarlyStopping:
    """
    Stops training when validation metric doesn't improve
    - Prevents overfitting
    - Saves training time
    """
    def __init__(self, patience: int = 8, metric: str = "val_loss", delta: float = 0.0):
        self.patience = patience
        self.metric = metric
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
    
    def __call__(self, current_score: float, epoch: int = 0) -> bool:
        """
        Returns True if training should stop
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            return False
        
        if current_score < (self.best_score - self.delta):
            self.best_score = current_score
            self.counter = 0
            self.best_epoch = epoch
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"⚠️  Early stopping at epoch {epoch}")
                print(f"   Best score was {self.best_score:.4f} at epoch {self.best_epoch}")
                return True
            return False


# ============================================
# 4. CHECKPOINT MANAGER
# ============================================

class CheckpointManager:
    """
    Manages model checkpoints
    - Saves best models
    - Keeps only N most recent checkpoints
    - Auto-removes old ones to save disk space
    """
    def __init__(self, model_name: str, max_keep: int = 3):
        self.model_name = model_name
        self.max_keep = max_keep
        self.checkpoint_dir = Path('./checkpoints') / model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints = []
    
    def save(self, epoch: int, step: int, model, optimizer, 
             metrics: Dict, is_best: bool = False) -> Path:
        """
        Save checkpoint
        
        Args:
            epoch: Current epoch
            step: Global step
            model: Model to save
            optimizer: Optimizer state
            metrics: Training metrics
            is_best: If True, save as best model
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'metrics': metrics,
        }
        
        filename = "best_model.pt" if is_best else f"checkpoint_ep{epoch}_step{step}.pt"
        path = self.checkpoint_dir / filename
        
        torch.save(checkpoint_data, path)
        
        # Track non-best checkpoints for cleanup
        if not is_best:
            self.checkpoints.append(path)
            
            # Remove old checkpoints
            if len(self.checkpoints) > self.max_keep:
                old_ckpt = self.checkpoints.pop(0)
                if old_ckpt.exists():
                    old_ckpt.unlink()
                    print(f"  🗑️  Removed old checkpoint: {old_ckpt.name}")
        
        file_size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  💾 Saved: {filename} ({file_size_mb:.1f}MB)")
        
        return path
    
    def load(self, model, optimizer, path=None) -> Tuple[int, int]:
        """
        Load checkpoint
        
        Returns:
            (epoch, step) from checkpoint
        """
        if path is None:
            path = self.checkpoint_dir / "best_model.pt"
        
        if not path.exists():
            print(f"⚠️  Checkpoint not found: {path}")
            return 0, 0
        
        device = next(model.parameters()).device
        checkpoint_data = torch.load(path, map_location=device)
        
        model.load_state_dict(checkpoint_data['model_state'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint_data['optimizer_state'])
        
        file_size_mb = path.stat().st_size / (1024 * 1024)
        print(f"✅ Loaded: {path.name} ({file_size_mb:.1f}MB)")
        
        return checkpoint_data['epoch'], checkpoint_data['step']


# ============================================
# 5. DATALOADER OPTIMIZATION
# ============================================

class DataLoaderOptimizer:
    """
    Recommendations for DataLoader optimization
    """
    
    @staticmethod
    def get_optimized_config(dataset_size: int, num_gpus: int = 1) -> Dict:
        """
        Get optimized DataLoader config based on dataset size
        
        Args:
            dataset_size: Number of samples
            num_gpus: Number of GPUs available
        
        Returns:
            Dictionary with optimal config
        """
        # Heuristics for optimal settings
        if dataset_size < 10000:
            batch_size = 32
            num_workers = 2
        elif dataset_size < 100000:
            batch_size = 64
            num_workers = 4
        else:
            batch_size = 128
            num_workers = 8
        
        return {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'pin_memory': True,  # Copy data directly to GPU
            'prefetch_factor': 2,  # Prefetch 2 batches
            'persistent_workers': True,  # Keep workers alive
            'drop_last': True,  # Drop incomplete batch
        }


# ============================================
# 6. MODEL PRUNING
# ============================================

class ModelPruner:
    """
    Model pruning for compression
    - L1 unstructured pruning: removes individual weights
    - Reduces model size by N% with minimal accuracy loss
    """
    
    @staticmethod
    def apply_l1_pruning(model: nn.Module, amount: float = 0.3) -> nn.Module:
        """
        Apply L1 unstructured pruning
        
        Args:
            model: Model to prune
            amount: Fraction of weights to remove (0.0-1.0)
        
        Returns:
            Pruned model
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.l1_unstructured(module, name='weight', amount=amount)
                # Remove mask to make pruning permanent
                prune.remove(module, 'weight')
        
        return model
    
    @staticmethod
    def apply_structured_pruning(model: nn.Module, amount: float = 0.3) -> nn.Module:
        """
        Apply structured pruning (remove entire filters/channels)
        More efficient for inference than unstructured pruning
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(
                    module, 
                    name='weight', 
                    amount=amount, 
                    n=2,  # Prune by channels
                    dim=0   # Prune output channels
                )
                prune.remove(module, 'weight')
        
        return model
    
    @staticmethod
    def get_sparsity(model: nn.Module) -> Dict:
        """
        Calculate model sparsity (percentage of zeros)
        """
        total = 0
        zeros = 0
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                total += param.numel()
                zeros += (param == 0).sum().item()
        
        sparsity = (zeros / total * 100) if total > 0 else 0
        
        return {
            'total_weights': total,
            'zero_weights': zeros,
            'sparsity_percent': sparsity,
            'compression_ratio': total / (total - zeros) if (total - zeros) > 0 else 1.0
        }


# ============================================
# 7. COMPLETE OPTIMIZATION SUMMARY
# ============================================

class OptimizationSummary:
    """
    Summary of all optimizations and their expected benefits
    """
    
    TECHNIQUES = {
        "AMP": {
            "memory_reduction": "40-50%",
            "speed_improvement": "1.5-2x",
            "implementation": "torch.cuda.amp.autocast + GradScaler"
        },
        "Gradient Checkpointing": {
            "memory_reduction": "30-40%",
            "speed_improvement": "-5-10% (slower but worth it)",
            "implementation": "torch.utils.checkpoint.checkpoint"
        },
        "Early Stopping": {
            "memory_reduction": "0%",
            "speed_improvement": "10-30% training time saved",
            "implementation": "Monitor val_loss, stop if no improvement"
        },
        "DataLoader Optimization": {
            "memory_reduction": "0%",
            "speed_improvement": "2-3x data loading",
            "implementation": "num_workers, pin_memory, prefetch"
        },
        "Model Pruning": {
            "memory_reduction": "30% (configurable)",
            "speed_improvement": "10-20% inference",
            "implementation": "L1 unstructured or structured pruning"
        },
    }
    
    @staticmethod
    def print_summary():
        """Print summary table"""
        print("\n" + "="*100)
        print("OPTIMIZATION TECHNIQUES SUMMARY")
        print("="*100)
        
        for technique, details in OptimizationSummary.TECHNIQUES.items():
            print(f"\n✅ {technique}")
            print(f"   Memory: {details['memory_reduction']}")
            print(f"   Speed: {details['speed_improvement']}")
            print(f"   How: {details['implementation']}")
        
        print("\n" + "="*100)
        print("EXPECTED OVERALL IMPROVEMENT: 2-3x faster, 40-50% less VRAM")
        print("="*100 + "\n")


# ============================================
# UTILITIES
# ============================================

def cleanup_gpu_memory():
    """Clean up GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"💾 GPU Memory: {allocated:.1f}GB / {total:.1f}GB")


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB (float32)"""
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / (1024 * 1024)
    return size_mb


if __name__ == "__main__":
    OptimizationSummary.print_summary()
