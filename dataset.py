import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import RandAugment
import numpy as np
from PIL import Image
import pandas as pd
import os

class SkinCancerDataset(Dataset):
    def __init__(self, image_dir, metadata_file=None, image_size=224, 
                 augmentation=False, augmentation_strength=0.5, split="train"):
        self.image_dir = image_dir
        self.image_size = image_size
        self.split = split
        self.augmentation = augmentation and split == "train"
        
        # Load image paths and labels
        self.images = []
        self.labels = []
        self.clinical_info = []
        
        # Assume directory structure: class_name/image_name.jpg
        for class_idx, class_name in enumerate(sorted(os.listdir(image_dir))):
            class_path = os.path.join(image_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.images.append(os.path.join(class_path, img_name))
                        self.labels.append(class_idx)
        
        # Load clinical metadata if provided
        if metadata_file and os.path.exists(metadata_file):
            self.metadata = pd.read_csv(metadata_file)
        else:
            self.metadata = None
        
        # Transform
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        if self.augmentation:
            self.aug_transform = transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                RandAugment(num_ops=int(2 * augmentation_strength), magnitude=int(9 * augmentation_strength)),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ])
        else:
            self.aug_transform = None
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.images[idx]).convert('RGB')
        
        # Augmentation
        if self.aug_transform:
            image = self.aug_transform(image)
        
        # Base transform
        image = self.base_transform(image)
        label = self.labels[idx]
        
        # Get clinical info if available
        clinical_data = self._get_clinical_info(idx)
        
        return {
            'image': image,
            'label': label,
            'clinical_info': clinical_data,
            'image_path': self.images[idx]
        }
    
    def _get_clinical_info(self, idx):
        if self.metadata is None:
            return torch.zeros(3)  # age, gender, history
        
        # Placeholder: extract age, gender, history
        # Adjust based on your metadata structure
        age = torch.tensor([0.0], dtype=torch.float32)
        gender = torch.tensor([0.0], dtype=torch.float32)  # 0: male, 1: female
        history = torch.tensor([0.0], dtype=torch.float32)
        
        return torch.cat([age, gender, history])

def get_dataloaders(config, num_workers=4, pin_memory=True):
    train_dataset = SkinCancerDataset(
        config.train_path,
        image_size=config.image_size,
        augmentation=config.augmentation,
        split="train"
    )
    val_dataset = SkinCancerDataset(
        config.val_path,
        image_size=config.image_size,
        augmentation=False,
        split="val"
    )
    test_dataset = SkinCancerDataset(
        config.test_path,
        image_size=config.image_size,
        augmentation=False,
        split="test"
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader
