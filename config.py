from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class ModelConfig:
    name: str
    architecture: str
    pretrained: bool = True
    num_classes: int = 7
    input_size: int = 224

@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    
    # Memory & Checkpointing
    accumulation_steps: int = 4
    save_checkpoint_interval: int = 500  # steps
    clean_old_checkpoints: bool = True
    max_checkpoint_keep: int = 3
    
    # Early Stopping
    early_stopping_patience: int = 15
    early_stopping_metric: str = "val_loss"
    
    # Loss & Optimization
    loss_fn: str = "focal"  # "dice" or "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    dice_smooth: float = 1e-5
    
    # Augmentation
    augmentation: bool = True
    augmentation_strength: float = 0.5

@dataclass
class DataConfig:
    train_path: str
    val_path: str
    test_path: str
    num_workers: int = 4
    pin_memory: bool = True
    image_size: int = 224
    
    # Multimodal info
    use_clinical_info: bool = True
    clinical_features: List[str] = field(default_factory=lambda: ["age", "gender", "history"])

@dataclass
class VisualizationConfig:
    use_gradcam: bool = True
    use_tsne: bool = True
    tsne_sample_size: int = 500
    save_interval: int = 5  # epochs

# Model Configurations
MODELS_TO_TRAIN = [
    ModelConfig(name="resnet50", architecture="resnet50"),
    ModelConfig(name="resnet101", architecture="resnet101"),
    ModelConfig(name="inception_v3", architecture="inception_v3", input_size=299),
    ModelConfig(name="densenet121", architecture="densenet121"),
    ModelConfig(name="efficientnet_b0", architecture="efficientnet_b0", input_size=224),
    ModelConfig(name="efficientnet_b1", architecture="efficientnet_b1", input_size=240),
    ModelConfig(name="efficientnet_b2", architecture="efficientnet_b2", input_size=260),
    ModelConfig(name="efficientnet_b3", architecture="efficientnet_b3", input_size=300),
    ModelConfig(name="efficientnet_b4", architecture="efficientnet_b4", input_size=380),
    ModelConfig(name="efficientnet_b5", architecture="efficientnet_b5", input_size=456),
    ModelConfig(name="efficientnet_b6", architecture="efficientnet_b6", input_size=528),
    ModelConfig(name="efficientnet_b7", architecture="efficientnet_b7", input_size=600),
    ModelConfig(name="regnet_y_400mf", architecture="regnet_y_400mf"),
    ModelConfig(name="regnet_y_8gf", architecture="regnet_y_8gf"),
    ModelConfig(name="regnet_x_400mf", architecture="regnet_x_400mf"),
    ModelConfig(name="regnet_x_8gf", architecture="regnet_x_8gf"),
    ModelConfig(name="vit_b_16", architecture="vit_b_16", input_size=224),
    ModelConfig(name="vit_l_16", architecture="vit_l_16", input_size=224),
    ModelConfig(name="swin_b", architecture="swin_b"),
    ModelConfig(name="coatnet_0", architecture="coatnet_0"),
]

DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_DATA_CONFIG = DataConfig(
    train_path="/Users/hongviet/Documents/GitHub/Skin_Cancer_Recognition/data/train",
    val_path="/Users/hongviet/Documents/GitHub/Skin_Cancer_Recognition/data/val",
    test_path="/Users/hongviet/Documents/GitHub/Skin_Cancer_Recognition/data/test",
)
DEFAULT_VIZ_CONFIG = VisualizationConfig()
