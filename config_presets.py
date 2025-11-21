"""
Configuration presets for different hardware/scenarios
Use these as starting points, then tune based on your specific setup
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class HardwareProfile:
    """Hardware-specific configurations"""
    
    # GPU Configuration
    gpu_vram_gb: int  # Available VRAM in GB
    num_gpus: int = 1
    gpu_type: str = "unknown"  # V100, P100, A100, RTX3090, etc.
    
    # CPU Configuration
    num_cpu_cores: int = 8
    cpu_memory_gb: int = 32
    
    # Storage
    ssd_available: bool = True


class ConfigPresets:
    """Pre-configured settings for different scenarios"""
    
    # ============================================
    # NVIDIA P100 (16GB VRAM)
    # ============================================
    P100 = {
        "batch_size": 64,
        "num_workers": 4,
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": True,
        
        # Optimizations
        "use_mixed_precision": True,
        "use_gradient_checkpointing": True,
        "enable_pruning": True,
        "pruning_amount": 0.3,
        
        # Training
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "early_stopping_patience": 8,
        "max_checkpoint_keep": 3,
        
        # Model
        "input_size": 224,
        
        "description": "P100 (16GB) - NVIDIA Tesla P100",
        "expected_memory_usage": "8-10 GB",
        "expected_training_time": "4-6 hours for full dataset"
    }
    
    # ============================================
    # NVIDIA V100 (32GB VRAM)
    # ============================================
    V100 = {
        "batch_size": 128,
        "num_workers": 6,
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": True,
        
        "use_mixed_precision": True,
        "use_gradient_checkpointing": False,  # V100 has enough memory
        "enable_pruning": True,
        "pruning_amount": 0.3,
        
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "early_stopping_patience": 8,
        "max_checkpoint_keep": 3,
        
        "input_size": 224,
        
        "description": "V100 (32GB) - NVIDIA Tesla V100",
        "expected_memory_usage": "16-20 GB",
        "expected_training_time": "2-3 hours for full dataset"
    }
    
    # ============================================
    # NVIDIA A100 (40GB VRAM)
    # ============================================
    A100 = {
        "batch_size": 256,
        "num_workers": 8,
        "pin_memory": True,
        "prefetch_factor": 3,
        "persistent_workers": True,
        
        "use_mixed_precision": True,
        "use_gradient_checkpointing": False,  # Not needed
        "enable_pruning": True,
        "pruning_amount": 0.3,
        
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "early_stopping_patience": 8,
        "max_checkpoint_keep": 3,
        
        "input_size": 224,
        
        "description": "A100 (40GB) - NVIDIA A100",
        "expected_memory_usage": "20-25 GB",
        "expected_training_time": "1-2 hours for full dataset"
    }
    
    # ============================================
    # RTX 4090 (24GB VRAM)
    # ============================================
    RTX_4090 = {
        "batch_size": 96,
        "num_workers": 8,
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": True,
        
        "use_mixed_precision": True,
        "use_gradient_checkpointing": False,
        "enable_pruning": True,
        "pruning_amount": 0.3,
        
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "early_stopping_patience": 8,
        "max_checkpoint_keep": 3,
        
        "input_size": 224,
        
        "description": "RTX 4090 (24GB) - Consumer GPU",
        "expected_memory_usage": "12-15 GB",
        "expected_training_time": "2-3 hours for full dataset"
    }
    
    # ============================================
    # RTX 3090 (24GB VRAM)
    # ============================================
    RTX_3090 = {
        "batch_size": 64,
        "num_workers": 6,
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": True,
        
        "use_mixed_precision": True,
        "use_gradient_checkpointing": True,
        "enable_pruning": True,
        "pruning_amount": 0.3,
        
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "early_stopping_patience": 8,
        "max_checkpoint_keep": 3,
        
        "input_size": 224,
        
        "description": "RTX 3090 (24GB) - Consumer GPU",
        "expected_memory_usage": "12-15 GB",
        "expected_training_time": "3-4 hours for full dataset"
    }
    
    # ============================================
    # CPU-ONLY (Memory-constrained)
    # ============================================
    CPU_ONLY = {
        "batch_size": 16,
        "num_workers": 2,
        "pin_memory": False,
        "prefetch_factor": 1,
        "persistent_workers": False,
        
        "use_mixed_precision": False,
        "use_gradient_checkpointing": False,
        "enable_pruning": False,
        
        "num_epochs": 5,  # Only few epochs for CPU testing
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "early_stopping_patience": 3,
        "max_checkpoint_keep": 1,
        
        "input_size": 192,  # Smaller input
        
        "description": "CPU-ONLY - For testing purposes",
        "expected_memory_usage": "4-6 GB RAM",
        "expected_training_time": "Very slow, for testing only"
    }
    
    # ============================================
    # MOBILE/EDGE (Colab Free)
    # ============================================
    COLAB_FREE = {
        "batch_size": 32,
        "num_workers": 2,
        "pin_memory": True,
        "prefetch_factor": 1,
        "persistent_workers": False,
        
        "use_mixed_precision": True,
        "use_gradient_checkpointing": True,
        "enable_pruning": True,
        "pruning_amount": 0.3,
        
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "early_stopping_patience": 5,
        "max_checkpoint_keep": 1,
        
        "input_size": 192,  # Smaller for speed
        
        "description": "Google Colab Free (K80, 12GB)",
        "expected_memory_usage": "6-8 GB",
        "expected_training_time": "6-8 hours for full dataset"
    }
    
    # ============================================
    # COLAB PRO
    # ============================================
    COLAB_PRO = {
        "batch_size": 64,
        "num_workers": 4,
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": True,
        
        "use_mixed_precision": True,
        "use_gradient_checkpointing": True,
        "enable_pruning": True,
        "pruning_amount": 0.3,
        
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "early_stopping_patience": 8,
        "max_checkpoint_keep": 2,
        
        "input_size": 224,
        
        "description": "Google Colab Pro (V100, 32GB)",
        "expected_memory_usage": "12-16 GB",
        "expected_training_time": "2-3 hours for full dataset"
    }


def get_config_for_gpu(gpu_name: str) -> dict:
    """Get recommended config for a specific GPU"""
    gpu_name = gpu_name.upper().strip()
    
    config_map = {
        "P100": ConfigPresets.P100,
        "V100": ConfigPresets.V100,
        "A100": ConfigPresets.A100,
        "RTX 4090": ConfigPresets.RTX_4090,
        "RTX 3090": ConfigPresets.RTX_3090,
        "K80": ConfigPresets.COLAB_FREE,
        "CPU": ConfigPresets.CPU_ONLY,
        "COLAB": ConfigPresets.COLAB_PRO,
    }
    
    # Try to find matching GPU
    for key, config in config_map.items():
        if key in gpu_name:
            return config
    
    # Default to V100 if not found
    print(f"⚠️  GPU '{gpu_name}' not found, using V100 preset as default")
    return ConfigPresets.V100


def print_config_recommendations():
    """Print all available configurations"""
    print("\n" + "="*100)
    print("CONFIGURATION PRESETS FOR DIFFERENT HARDWARE")
    print("="*100)
    
    configs = {
        "P100": ConfigPresets.P100,
        "V100": ConfigPresets.V100,
        "A100": ConfigPresets.A100,
        "RTX 4090": ConfigPresets.RTX_4090,
        "RTX 3090": ConfigPresets.RTX_3090,
        "CPU-ONLY": ConfigPresets.CPU_ONLY,
        "COLAB FREE": ConfigPresets.COLAB_FREE,
        "COLAB PRO": ConfigPresets.COLAB_PRO,
    }
    
    for name, config in configs.items():
        print(f"\n{'='*100}")
        print(f"🖥️  {name}")
        print(f"{'='*100}")
        print(f"Description: {config['description']}")
        print(f"\nKey Settings:")
        print(f"  • Batch Size: {config['batch_size']}")
        print(f"  • Num Workers: {config['num_workers']}")
        print(f"  • Pin Memory: {config['pin_memory']}")
        print(f"  • AMP Enabled: {config['use_mixed_precision']}")
        print(f"  • Gradient Checkpointing: {config['use_gradient_checkpointing']}")
        print(f"  • Pruning Enabled: {config['enable_pruning']}")
        print(f"\nExpected Performance:")
        print(f"  • Memory Usage: {config['expected_memory_usage']}")
        print(f"  • Training Time: {config['expected_training_time']}")


# ============================================
# HELPER FUNCTIONS
# ============================================

def detect_gpu_type() -> str:
    """Detect GPU type and return appropriate preset"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ Detected GPU: {gpu_name}")
            return get_config_for_gpu(gpu_name)
        else:
            print("⚠️  CUDA not available, using CPU preset")
            return ConfigPresets.CPU_ONLY
    except Exception as e:
        print(f"❌ Error detecting GPU: {e}")
        return ConfigPresets.V100  # Safe default


def auto_tune_batch_size(available_vram_gb: int) -> int:
    """Auto-tune batch size based on available VRAM"""
    if available_vram_gb < 4:
        return 8
    elif available_vram_gb < 8:
        return 16
    elif available_vram_gb < 16:
        return 32
    elif available_vram_gb < 24:
        return 64
    elif available_vram_gb < 32:
        return 96
    else:
        return 128


if __name__ == "__main__":
    print_config_recommendations()
    
    print("\n" + "="*100)
    print("AUTO-DETECTION")
    print("="*100)
    config = detect_gpu_type()
    print("\n✅ Selected config:")
    for key, value in list(config.items())[:5]:
        print(f"  • {key}: {value}")
