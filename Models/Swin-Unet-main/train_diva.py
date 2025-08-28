import argparse
import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from networks.vision_transformer import SwinUnet as ViT_seg
from datasets_diva.dataset_divahisdb import DivaHisDB_dataset
from utils import DiceLoss


def get_model(args, config):
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    try:
        net.load_from(config)
    except Exception:
        pass
    return net


def train(args):
    # basic defaults
    args.patch_size = args.patch_size or 224
    args.batch_size = args.batch_size or 8
    args.base_lr = args.base_lr or 1e-4
    args.num_workers = args.num_workers or 4
    args.max_epochs = args.max_epochs or 100

    # config placeholder (reuse existing config machinery)
    from config import get_config
    config = get_config(args)

    # dataset
    train_ds = DivaHisDB_dataset(args.diva_root, manuscript=args.manuscript, split='training', patch_size=args.patch_size, augment=True)
    val_ds = DivaHisDB_dataset(args.diva_root, manuscript=args.manuscript, split='validation', patch_size=args.patch_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    net = get_model(args, config)
    if args.n_gpu > 1:
        net = nn.DataParallel(net)

    # losses: BCEWithLogits + Dice (expects logits->sigmoid inside dice?)
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss(num_classes=args.num_classes)
    optimizer = optim.AdamW(net.parameters(), lr=args.base_lr, weight_decay=0.01)

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.max_epochs):
        net.train()
        train_loss = 0.0
        for batch in train_loader:
            imgs = batch['image'].cuda()
            labels = batch['label'].cuda()
            outputs = net(imgs)
            # outputs: (B, C, H, W); labels: (B, C, H, W) float
            loss_bce = bce(outputs, labels)
            # for DiceLoss we need class indices; approximate by argmax across channels
            try:
                loss_dice = dice(outputs, labels.argmax(dim=1), softmax=False)
            except Exception:
                loss_dice = torch.tensor(0.0).cuda()
            loss = 0.5 * loss_bce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print(f'Epoch {epoch+1}/{args.max_epochs} Train loss: {train_loss:.4f}')

        # quick validation (compute loss only)
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].cuda()
                labels = batch['label'].cuda()
                outputs = net(imgs)
                loss_bce = bce(outputs, labels)
                try:
                    loss_dice = dice(outputs, labels.argmax(dim=1), softmax=False)
                except Exception:
                    loss_dice = torch.tensor(0.0).cuda()
                loss = 0.5 * loss_bce + 0.5 * loss_dice
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))
        print(f'Epoch {epoch+1}/{args.max_epochs} Val loss: {val_loss:.4f}')

        # save checkpoint
        torch.save(net.state_dict(), os.path.join(args.output_dir, f'epoch_{epoch+1}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--diva_root', type=str, required=True)
    parser.add_argument('--manuscript', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='diva_train_out')
    parser.add_argument('--patch_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--n_gpu', type=int, default=max(1, torch.cuda.device_count()))
    args = parser.parse_args()
    train(args)
