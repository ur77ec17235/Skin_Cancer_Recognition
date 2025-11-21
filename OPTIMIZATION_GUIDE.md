# Advanced Training Optimizations for Skin Cancer Recognition

## 🚀 Overview

Notebook của bạn đã được cải thiện với 6 kỹ thuật tối ưu hóa chính:

1. **AMP (Automatic Mixed Precision)** - Giảm VRAM 50%, tăng tốc 2x
2. **Gradient Checkpointing** - Giảm memory 30-40%
3. **Early Stopping** - Dừng training sớm, tiết kiệm thời gian
4. **Checkpoint Manager** - Lưu best models, tự xóa old checkpoints
5. **DataLoader Optimization** - Tăng throughput 2-3x
6. **Model Pruning** - Giảm model size 30%, fast inference

---

## 📊 Expected Performance Gains

| Technique | Memory Savings | Speed Improvement | Trade-offs |
|-----------|---|---|---|
| AMP | 40-50% | 1.5-2x | None, pure win |
| Gradient Checkpointing | 30-40% | -5-10% | Slower, but memory worth it |
| DataLoader Optimization | 0% | 2-3x | More CPU usage |
| Model Pruning | 30% | 10-20% inference | Slight accuracy loss |
| **COMBINED** | **50-60%** | **2-3x** | Optimal for GPU |

---

## 🔧 Implementation Details

### 1️⃣ AMP (Automatic Mixed Precision)

**Cách hoạt động:**
- Dùng `float16` cho forward/backward pass
- Giữ `float32` cho weights
- GradScaler tự động scale loss để tránh underflow

**Code:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast(device_type='cuda'):
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
scaler.step(optimizer)
scaler.update()
```

**Lợi ích:**
- ✅ 50% giảm VRAM
- ✅ 1.5-2x tăng tốc
- ✅ Không ảnh hưởng độ chính xác

---

### 2️⃣ Gradient Checkpointing

**Cách hoạt động:**
- Không cache activations trong forward pass
- Recompute chúng trong backward pass
- Trade memory vs compute

**Code:**
```python
import torch.utils.checkpoint as checkpoint

def forward(self, x):
    if self.use_checkpoint and self.training:
        return checkpoint.checkpoint(
            self._forward_impl,
            x,
            use_reentrant=False
        )
```

**Lợi ích:**
- ✅ 30-40% giảm memory
- ✅ Cho phép batch size lớn hơn
- ⚠️ Slow ~5-10%

---

### 3️⃣ Early Stopping

**Cách hoạt động:**
- Giám sát `val_loss` mỗi epoch
- Nếu không cải thiện trong N epochs → dừng
- Lưu best model

**Code:**
```python
early_stopping = EarlyStopping(patience=8)

for epoch in range(num_epochs):
    # training...
    val_loss = validate(model, val_loader)
    
    if early_stopping(val_loss):
        print("Stop training!")
        break
```

**Lợi ích:**
- ✅ Tiết kiệm 10-30% training time
- ✅ Tự động prevent overfitting
- ✅ Lưu best weights tự động

---

### 4️⃣ Checkpoint Manager

**Cách hoạt động:**
- Lưu best model sau mỗi epoch
- Giữ tối đa N phiên bản gần đây
- Tự xóa old checkpoints để save disk

**Code:**
```python
ckpt_manager = CheckpointManager(model_name, max_keep=3)

# Save checkpoint
if val_loss < best_loss:
    ckpt_manager.save(epoch, step, model, optimizer, metrics, is_best=True)

# Load checkpoint
epoch, step = ckpt_manager.load(model, optimizer)
```

**Lợi ích:**
- ✅ Không lo mất best model
- ✅ Tiết kiệm disk space (chỉ keep 3)
- ✅ Phục hồi nhanh khi crash

---

### 5️⃣ DataLoader Optimization

**Cách hoạt động:**
- `num_workers=4`: load data song song trên 4 threads
- `pin_memory=True`: pin data → GPU trực tiếp (không qua CPU)
- `prefetch_factor=2`: prefetch 2 batches trước
- `persistent_workers=True`: keep workers alive

**Code:**
```python
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,           # ✅ Parallel loading
    pin_memory=True,         # ✅ Direct to GPU
    prefetch_factor=2,       # ✅ Prefetch batches
    persistent_workers=True  # ✅ Keep workers
)
```

**Lợi ích:**
- ✅ 2-3x tăng throughput
- ✅ GPU không idle chờ data
- ✅ Tối thiểu overhead từ I/O

---

### 6️⃣ Model Pruning

**Cách hoạt động:**
- **Unstructured pruning**: xóa weights riêng lẻ theo magnitude
- **Structured pruning**: xóa toàn bộ filters/channels
- Áp dụng sau khi train, trước khi deploy

**Code:**
```python
from torch.nn.utils import prune

# Unstructured
for module in model.modules():
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        prune.l1_unstructured(module, 'weight', amount=0.3)
        prune.remove(module, 'weight')  # Make permanent

# Check sparsity
sparsity = 100 * (weights == 0).sum() / weights.numel()
```

**Lợi ích:**
- ✅ 30% giảm model size
- ✅ 10-20% inference tăng tốc
- ✅ Phù hợp cho mobile/edge

---

## 📝 Usage in Notebook

### Minimal Example:
```python
# Setup
from torch.cuda.amp import autocast, GradScaler
from training_optimizations import CheckpointManager, EarlyStopping

scaler = GradScaler()
ckpt_manager = CheckpointManager("my_model")
early_stopping = EarlyStopping(patience=8)

# Training loop
for epoch in range(num_epochs):
    train_loss = 0
    for batch in train_loader:
        with autocast(device_type='cuda'):
            outputs = model(batch)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    # Validation
    val_loss = validate()
    
    # Checkpoint
    if val_loss < best_val_loss:
        ckpt_manager.save(epoch, step, model, optimizer, 
                         {'val_loss': val_loss}, is_best=True)
    
    # Early stopping
    if early_stopping(val_loss):
        break
```

---

## 📈 Recommended Config for P100

```python
TrainingConfig(
    batch_size=64,  # P100 có 16GB, dùng 64 khá ổn
    num_epochs=50,
    learning_rate=1e-3,
    weight_decay=1e-4,
    use_mixed_precision=True,      # ✅ Bắt buộc
    use_gradient_checkpointing=True,# ✅ Để fit batch size lớn
    num_workers=4,                  # ✅ Song song load
    pin_memory=True,                # ✅ GPU-direct copy
    early_stopping_patience=8,      # ✅ Tự động stop
    enable_pruning=True,            # ✅ Giảm size sau
    pruning_amount=0.3,             # ✅ Xóa 30% weights
)
```

---

## ⚠️ Common Issues & Solutions

### Issue: "CUDA out of memory"
**Solutions:**
1. Giảm batch_size
2. Enable gradient checkpointing
3. Enable AMP
4. Giảm input size (224 → 200)

### Issue: "Training very slow"
**Solutions:**
1. Increase `num_workers`
2. Tắt gradient checkpointing (nó chậm ~5-10%)
3. Giảm model complexity
4. Kiểm tra `torch.cuda.memory_allocated()` có memory leak?

### Issue: "Model accuracy thấp sau pruning"
**Solutions:**
1. Giảm `pruning_amount` (0.3 → 0.15)
2. Fine-tune sau pruning
3. Dùng structured pruning thay vì unstructured

---

## 🎯 Checkpoints & Outputs

Notebook sẽ tạo:

```
./checkpoints/
├── model_name/
│   ├── best_model.pt           # ✅ Best validation model
│   ├── best_model_pruned.pt    # ✅ Pruned version
│   └── checkpoint_epoch*.pt    # Recent checkpoints (keep 3)

./results/
├── training_results.json       # Loss, accuracy history
├── training_statistics.csv     # Time statistics
├── model_compression.csv       # Pruning analysis
└── optimization_report.json    # Complete report

./visualizations/
├── training_history.png        # Loss/Accuracy curves
├── training_speed_per_epoch.png
├── model_gradcam_*.png
└── model_tsne.png
```

---

## 📚 Key Metrics to Monitor

```python
# Per epoch, log:
{
    'train_loss': float,           # Training loss
    'train_acc': float,            # Training accuracy
    'val_loss': float,             # Validation loss
    'val_acc': float,              # Validation accuracy
    'epoch_time': float,           # Seconds per epoch
    'lr': float,                   # Learning rate
    'gpu_memory_mb': float,        # GPU memory used
}
```

---

## 🚀 Quick Start

1. **Load notebook:**
   ```
   notebook1f93529628.ipynb
   ```

2. **Run cells in order:**
   - Import + Setup
   - Config + Loss functions
   - Dataset + DataLoader
   - Model definition
   - Training loop (optimized)
   - Evaluation + Pruning
   - Visualization + Report

3. **Check results:**
   ```
   ls ./checkpoints/*/best_model_pruned.pt
   cat ./results/optimization_report.json
   ```

---

## 📖 References

- [PyTorch AMP](https://pytorch.org/docs/stable/amp.html)
- [Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)
- [Model Pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [Efficient DataLoading](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

## ✅ Optimization Checklist

- [ ] AMP enabled (`use_mixed_precision=True`)
- [ ] Gradient checkpointing enabled
- [ ] DataLoader optimized (num_workers, pin_memory)
- [ ] Early stopping configured
- [ ] Checkpoint manager setup
- [ ] Model pruning enabled
- [ ] Memory usage monitored
- [ ] Training speed improved 2-3x
- [ ] Model size reduced 30%
- [ ] Report generated

---

**Author:** Optimization Pipeline  
**Last Updated:** 2025-11-17  
**P100 Friendly:** ✅ Yes
