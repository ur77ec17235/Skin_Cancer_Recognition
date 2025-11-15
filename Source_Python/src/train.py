import os
import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import roc_auc_score

from dataset import DermDataset
from models import get_model
from losses import ComboLoss
from utils import set_seed, save_checkpoint, EarlyStopping
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(is_train=True, size=224):
    if is_train:
        return A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.4),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.5),
            A.ColorJitter(p=0.4),
            A.Normalize(),
            ToTensorV2(),
        ])
    return A.Compose([A.Resize(size, size), A.Normalize(), ToTensorV2()])


def train_epoch(model, loader, opt, loss_fn, device, accum_steps=1):
    model.train()
    total_loss = 0.0
    for step, (imgs, metas, labels) in enumerate(loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        metas = metas.to(device) if metas is not None else None

        out = model(imgs, metas)
        loss = loss_fn(out, labels.unsqueeze(1)) if out.dim()==2 else loss_fn(out, labels)
        loss = loss / accum_steps
        loss.backward()
        if (step + 1) % accum_steps == 0:
            opt.step()
            opt.zero_grad()
        total_loss += loss.item() * accum_steps
    return total_loss / len(loader)


def valid_epoch(model, loader, loss_fn, device):
    model.eval()
    ys = []
    ps = []
    losses = 0.0
    with torch.no_grad():
        for imgs, metas, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            metas = metas.to(device) if metas is not None else None
            out = model(imgs, metas)
            loss = loss_fn(out, labels.unsqueeze(1)) if out.dim()==2 else loss_fn(out, labels)
            losses += loss.item()
            probs = torch.sigmoid(out).detach().cpu().numpy()
            ps.append(probs.reshape(-1))
            ys.append(labels.cpu().numpy().reshape(-1))
    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    auc = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else 0.0
    return losses / len(loader), auc


def main(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df = pd.read_csv(args.metadata)
    # expect column isic_id and benign_malignant
    meta_cols = args.meta_cols.split(',') if args.meta_cols else []

    train_df = df.sample(frac=1 - args.val_frac, random_state=args.seed)
    val_df = df.drop(train_df.index)

    train_ds = DermDataset(train_df, args.image_dir, meta_cols=meta_cols, transforms=get_transforms(True))
    val_ds = DermDataset(val_df, args.image_dir, meta_cols=meta_cols, transforms=get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = get_model(args.model, num_meta_features=len(meta_cols), out_features=1, pretrained=True)
    model = model.to(device)

    opt = Adam(model.parameters(), lr=args.lr)
    loss_fn = ComboLoss()

    early = EarlyStopping(patience=args.patience)

    global_step = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, opt, loss_fn, device, accum_steps=args.accum_steps)
        val_loss, val_auc = valid_epoch(model, val_loader, loss_fn, device)
        t1 = time.time()
        print(f"Epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f} time={(t1-t0):.1f}s")

        # Save checkpoints per epoch (or you can change to per N steps)
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt_state': opt.state_dict(),
            'val_auc': val_auc,
        }
        save_checkpoint(ckpt, args.ckpt_dir, step=epoch, keep_last=args.keep_last)

        early.step(val_auc)
        if early.should_stop:
            print('Early stopping triggered')
            break

    print('Training finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, default='./all-isic-data-20240629/metadata.csv')
    parser.add_argument('--image_dir', type=str, default='./all-isic-data-20240629/images/')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--meta_cols', type=str, default='patient_age,sex')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--keep_last', type=int, default=3)
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--accum_steps', type=int, default=1)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()
    main(args)
