# Skin Cancer Recognition — PyTorch training utilities

This repo adds a PyTorch training pipeline to the existing notebook. It implements:

- Dataset supporting dermoscopy images + tabular patient metadata.
- Many backbones via `timm` (ResNet, EfficientNet B0-B7, DenseNet121, RegNet, ViT, Swin, CoAtNet, etc.)
- Image augmentation (albumentations)
- Losses: Dice + Focal (Combo)
- Checkpoint saving with keep-last-N behavior
- Early stopping
- Grad-CAM helper and t-SNE visualization
- Image + metadata fusion model

Files added:

- `src/dataset.py` — Dataset class (image + metadata)
- `src/models.py` — Wrapper using `timm` backbones plus meta-MLP fusion
- `src/losses.py` — Dice, Focal and Combo loss
- `src/utils.py` — checkpoint utilities and EarlyStopping
- `src/gradcam.py` — simple grad-cam via hooks
- `src/visualize.py` — t-SNE helper
- `src/train.py` — training CLI script
- `requirements.txt` — essential dependencies

Quick start (local, macOS):

1. Create a virtualenv and install requirements:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

2. Train (example):

    python src/train.py --model resnet50 --batch_size 16 --epochs 10 --meta_cols patient_age,sex

Notes / next steps:

- You can replace `--model resnet50` with `resnet101`, `inceptionv3`, `densenet121`, `efficientnet_b0`, `efficientnet_b7`, `vit_b16`, `vit_l16`, `swin`, `coatnet`, `regnety`, `regnetx`, etc. `timm` exposes many variants; check `timm.list_models()` for exact names.
- The training script uses AUROC on validation to decide early stopping and checkpointing.
- For Grad-CAM use the `src/gradcam.py` - pass target layer (e.g., model.backbone.blocks[-1] or model.backbone.layer4[-1]) depending on backbone.
- I kept the changes minimal and isolated from your notebook. If you want, I can convert the notebook to use these scripts and add example cells.
