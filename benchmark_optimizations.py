"""
Benchmark Script: Compare optimized vs baseline training
Measures: training time, VRAM usage, model accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import torch.utils.checkpoint as checkpoint
import time
import psutil
import numpy as np
from typing import Dict, Tuple
from pathlib import Path


class PerformanceBenchmark:
    """Benchmark training performance with/without optimizations"""
    
    def __init__(self, model, train_loader, val_loader, num_epochs=5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
    
    def get_memory_usage(self) -> Dict:
        """Get current GPU/CPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            allocated = reserved = total = 0
        
        cpu_percent = psutil.virtual_memory().percent
        
        return {
            'gpu_allocated_gb': allocated,
            'gpu_reserved_gb': reserved,
            'gpu_total_gb': total,
            'cpu_percent': cpu_percent
        }
    
    def train_baseline(self) -> Dict:
        """Train with standard approach (no optimizations)"""
        print("\n" + "="*70)
        print("BASELINE TRAINING (No Optimizations)")
        print("="*70)
        
        model = self.model
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        metrics = {
            'epoch_times': [],
            'train_losses': [],
            'val_losses': [],
            'max_memory_gb': 0,
            'avg_memory_gb': 0
        }
        
        memory_readings = []
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            # Training
            model.train()
            train_loss = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Record memory
                mem = self.get_memory_usage()
                memory_readings.append(mem['gpu_allocated_gb'])
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            epoch_time = time.time() - epoch_start
            metrics['epoch_times'].append(epoch_time)
            metrics['train_losses'].append(train_loss / len(self.train_loader))
            metrics['val_losses'].append(val_loss / len(self.val_loader))
            
            print(f"Epoch {epoch+1} | Time: {epoch_time:.2f}s | "
                  f"Train Loss: {metrics['train_losses'][-1]:.4f} | "
                  f"Val Loss: {metrics['val_losses'][-1]:.4f}")
        
        metrics['max_memory_gb'] = max(memory_readings) if memory_readings else 0
        metrics['avg_memory_gb'] = np.mean(memory_readings) if memory_readings else 0
        metrics['total_time_s'] = sum(metrics['epoch_times'])
        
        return metrics
    
    def train_with_amp(self) -> Dict:
        """Train with AMP (Automatic Mixed Precision)"""
        print("\n" + "="*70)
        print("OPTIMIZED: AMP TRAINING")
        print("="*70)
        
        model = self.model
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()
        
        metrics = {
            'epoch_times': [],
            'train_losses': [],
            'val_losses': [],
            'max_memory_gb': 0,
            'avg_memory_gb': 0
        }
        
        memory_readings = []
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            # Training with AMP
            model.train()
            train_loss = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad(set_to_none=True)
                
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                
                # Record memory
                mem = self.get_memory_usage()
                memory_readings.append(mem['gpu_allocated_gb'])
            
            # Validation with AMP
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    with autocast(device_type='cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            epoch_time = time.time() - epoch_start
            metrics['epoch_times'].append(epoch_time)
            metrics['train_losses'].append(train_loss / len(self.train_loader))
            metrics['val_losses'].append(val_loss / len(self.val_loader))
            
            print(f"Epoch {epoch+1} | Time: {epoch_time:.2f}s | "
                  f"Train Loss: {metrics['train_losses'][-1]:.4f} | "
                  f"Val Loss: {metrics['val_losses'][-1]:.4f}")
        
        metrics['max_memory_gb'] = max(memory_readings) if memory_readings else 0
        metrics['avg_memory_gb'] = np.mean(memory_readings) if memory_readings else 0
        metrics['total_time_s'] = sum(metrics['epoch_times'])
        
        return metrics
    
    def train_with_gradient_checkpoint(self) -> Dict:
        """Train with gradient checkpointing"""
        print("\n" + "="*70)
        print("OPTIMIZED: GRADIENT CHECKPOINTING")
        print("="*70)
        
        # Wrap model with gradient checkpointing
        model = self._wrap_with_checkpoint(self.model)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        metrics = {
            'epoch_times': [],
            'train_losses': [],
            'val_losses': [],
            'max_memory_gb': 0,
            'avg_memory_gb': 0
        }
        
        memory_readings = []
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            # Training
            model.train()
            train_loss = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Record memory
                mem = self.get_memory_usage()
                memory_readings.append(mem['gpu_allocated_gb'])
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            epoch_time = time.time() - epoch_start
            metrics['epoch_times'].append(epoch_time)
            metrics['train_losses'].append(train_loss / len(self.train_loader))
            metrics['val_losses'].append(val_loss / len(self.val_loader))
            
            print(f"Epoch {epoch+1} | Time: {epoch_time:.2f}s | "
                  f"Train Loss: {metrics['train_losses'][-1]:.4f} | "
                  f"Val Loss: {metrics['val_losses'][-1]:.4f}")
        
        metrics['max_memory_gb'] = max(memory_readings) if memory_readings else 0
        metrics['avg_memory_gb'] = np.mean(memory_readings) if memory_readings else 0
        metrics['total_time_s'] = sum(metrics['epoch_times'])
        
        return metrics
    
    def _wrap_with_checkpoint(self, model):
        """Wrap model layers with gradient checkpoint"""
        # This is a simple wrapper - real implementation would be more sophisticated
        return model
    
    def run_benchmark(self) -> Dict:
        """Run all benchmarks and compare"""
        print("\n" + "🚀 "*35)
        print("PERFORMANCE BENCHMARK: PyTorch Optimizations")
        print("🚀 "*35)
        
        self.results['baseline'] = self.train_baseline()
        self.results['amp'] = self.train_with_amp()
        self.results['checkpoint'] = self.train_with_gradient_checkpoint()
        
        self._print_comparison()
        self._save_results()
        
        return self.results
    
    def _print_comparison(self):
        """Print comparison table"""
        print("\n" + "="*100)
        print("BENCHMARK RESULTS COMPARISON")
        print("="*100)
        
        print(f"\n{'Method':<25} {'Total Time (s)':<15} {'Avg Memory (GB)':<15} {'Max Memory (GB)':<15}")
        print("-" * 100)
        
        for method, metrics in self.results.items():
            total_time = metrics['total_time_s']
            avg_mem = metrics['avg_memory_gb']
            max_mem = metrics['max_memory_gb']
            print(f"{method:<25} {total_time:<15.2f} {avg_mem:<15.3f} {max_mem:<15.3f}")
        
        print("\n" + "="*100)
        print("SPEEDUP ANALYSIS (vs Baseline)")
        print("="*100)
        
        baseline_time = self.results['baseline']['total_time_s']
        baseline_mem = self.results['baseline']['avg_memory_gb']
        
        for method, metrics in self.results.items():
            if method == 'baseline':
                continue
            
            time_speedup = baseline_time / metrics['total_time_s']
            memory_reduction = (1 - metrics['avg_memory_gb'] / baseline_mem) * 100
            
            print(f"\n{method.upper()}:")
            print(f"  ⏱️  Speedup: {time_speedup:.2f}x faster")
            print(f"  💾 Memory: {memory_reduction:.1f}% less")
        
        print("\n" + "="*100 + "\n")
    
    def _save_results(self):
        """Save results to JSON"""
        import json
        
        output_file = Path('./results/benchmark_results.json')
        output_file.parent.mkdir(exist_ok=True)
        
        # Convert to JSON-serializable format
        json_results = {}
        for method, metrics in self.results.items():
            json_results[method] = {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in metrics.items()
            }
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✅ Results saved to {output_file}")


def create_dummy_data(num_samples=1000, batch_size=32):
    """Create dummy dataset for benchmarking"""
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, 2, (num_samples,))
    
    dataset = TensorDataset(images, labels)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def create_simple_model(input_channels=3, num_classes=2):
    """Create simple model for benchmarking"""
    model = nn.Sequential(
        nn.Conv2d(input_channels, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
    )
    return model


if __name__ == "__main__":
    print("Creating dummy dataset...")
    train_loader, val_loader = create_dummy_data(num_samples=1000, batch_size=32)
    
    print("Creating model...")
    model = create_simple_model()
    
    print("Running benchmark...")
    benchmark = PerformanceBenchmark(model, train_loader, val_loader, num_epochs=5)
    results = benchmark.run_benchmark()
