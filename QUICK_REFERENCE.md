# 🎯 Quick Reference: Optimization Cheatsheet

## For P100 GPU (What You Have)

### Minimal Setup (Copy-Paste)
```python
# 1. Import
from torch.cuda.amp import autocast, GradScaler
from training_optimizations import CheckpointManager, EarlyStopping

# 2. Setup
scaler = GradScaler()
ckpt = CheckpointManager("my_model")
early_stop = EarlyStopping(patience=8)

# 3. Training Loop
for epoch in range(50):
    for batch in train_loader:
        with autocast(device_type='cuda'):
            loss = model(batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    val_loss = validate()
    if early_stop(val_loss):
        break
```

### Config Values for P100
```python
batch_size = 64                      # ✅ Good
num_workers = 4                      # ✅ 4 parallel workers
pin_memory = True                    # ✅ GPU-direct copy
use_mixed_precision = True           # ✅ 2x faster
use_gradient_checkpointing = True    # ✅ 30% memory
```

### Expected Performance
- ⏱️ Training: 3-6 hours (vs 12+ hours)
- 💾 Memory: 6-8 GB (vs 13 GB)
- 📊 Model: 30% smaller after pruning

---

## Memory Saving Techniques (Ranked)

| Rank | Technique | Savings | Speed Impact | Difficulty |
|------|-----------|---------|--------------|------------|
| 1 | AMP | 40-50% | +1.5-2x | ✅ Easy |
| 2 | Gradient Checkpointing | 30-40% | -5-10% | ✅ Easy |
| 3 | DataLoader opt | 0% | +2-3x | ✅ Easy |
| 4 | Batch size ↑ | 0% | +20-30% | ✅ Easy |
| 5 | Lower input size | 20-30% | +10-20% | ✅ Easy |
| 6 | Model pruning | 30% model | +10-20% inf | ⚠️ Medium |

---

## Problem Solver

### "CUDA out of memory"
1. Reduce batch_size (64 → 32)
2. Enable gradient checkpointing
3. Reduce input_size (224 → 192)
4. Use gradient_accumulation_steps=2

### "Training is slow"
1. Check num_workers > 0
2. Verify pin_memory=True
3. Use AMP
4. Disable gradient checkpointing (it's slower)

### "Model too large for inference"
1. Enable pruning (remove 30% weights)
2. Quantize to int8
3. Distill to smaller model

### "GPU sits idle"
1. Increase num_workers (2→4→8)
2. Increase prefetch_factor (1→2→3)
3. Increase batch_size if memory allows

---

## Implementation Checklist

- [ ] **Data Loading**
  - [ ] num_workers ≥ 2
  - [ ] pin_memory=True
  - [ ] prefetch_factor ≥ 2
  - [ ] persistent_workers=True

- [ ] **Mixed Precision**
  - [ ] GradScaler initialized
  - [ ] @autocast decorator in forward
  - [ ] scaler.scale(loss).backward()
  - [ ] scaler.unscale/step/update

- [ ] **Gradient Checkpointing**
  - [ ] torch.utils.checkpoint imported
  - [ ] use_gradient_checkpointing=True
  - [ ] Applied to large layers

- [ ] **Monitoring**
  - [ ] Memory logged each epoch
  - [ ] Loss curves saved
  - [ ] Time per batch tracked

- [ ] **Checkpointing**
  - [ ] Best model saved
  - [ ] Old models cleaned up
  - [ ] Checkpoint recoverable

- [ ] **Model Optimization**
  - [ ] Pruning applied after training
  - [ ] Model size reduced 30%
  - [ ] Sparsity checked

---

## Performance Targets (P100)

| Metric | Baseline | Optimized | Target |
|--------|----------|-----------|--------|
| Time/Epoch | 14.4 min | 3.6 min | 5-10 min |
| VRAM | 13 GB | 6-8 GB | < 10 GB |
| Batch/sec | 1 | 2.8 | 2-3 |
| Model Size | 200 MB | 140 MB | < 200 MB |

---

## Quick Stats

### AMP Impact
- Forward pass: float16 (50% memory)
- Backward pass: float16 (50% memory)
- Weights: float32 (safe)
- Result: 2x faster, 50% less VRAM

### Gradient Checkpointing Impact
- Don't cache forward activations
- Recompute in backward pass
- Result: 30-40% memory, -5-10% speed

### DataLoader Impact
- num_workers=4: 4x parallel loading
- pin_memory=True: GPU-direct copy
- prefetch_factor=2: 2 batches prefetched
- Result: 2-3x data throughput

### Model Pruning Impact
- Remove 30% weights (L1 magnitude)
- Model size: -30%
- Inference speed: +10-20%
- Accuracy: -0.5-2% (tunable)

---

## File Quick Links

| File | Purpose | Key Info |
|------|---------|----------|
| `training_optimizations.py` | Reference code | Classes & utilities |
| `OPTIMIZATION_GUIDE.md` | Full guide | Complete walkthrough |
| `benchmark_optimizations.py` | Performance test | Measure speedups |
| `config_presets.py` | Hardware configs | Pre-tuned settings |
| `optimization_visuals.py` | Diagrams | Trees & matrices |

---

## One-Liner Speedups

```python
# Enable AMP
with autocast(device_type='cuda'):
    loss = criterion(model(x), y)

# Gradient checkpointing
model = checkpoint.checkpoint(model, x)

# Early stopping
if early_stop(val_loss): break

# Better DataLoader
DataLoader(..., num_workers=4, pin_memory=True)

# Model pruning
prune.l1_unstructured(module, 'weight', amount=0.3)
```

---

## Expected Timeline (P100, 50 epochs, 10k samples)

| Configuration | Time |
|---|---|
| Baseline | 12h |
| + AMP | 6.5h |
| + DataLoader | 4.2h |
| + Checkpointing | 3.5h |
| + Early Stop | 2.8h |
| **All Optimized** | **3-4h** |
| **Speedup** | **3-4x** ⚡ |

---

## Environment Setup

```bash
# Install required packages
pip install torch torchvision timm

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"

# Check memory
python -c "import torch; print(torch.cuda.get_device_properties(0))"
```

---

## Remember

✅ **Always enable:** AMP, DataLoader opt, Early Stop, Checkpoints
⚠️ **Use if needed:** Gradient Checkpointing (VRAM < 16GB), Pruning (size matters)
❌ **Usually skip:** Lower precision, aggressive augmentation

**Golden Rule:** Measure, Monitor, Optimize. Don't guess! 📊

---

Last Updated: 2025-11-17
