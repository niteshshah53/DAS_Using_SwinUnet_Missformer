import logging
import os
import sys
from matplotlib import transforms
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from utils import DiceLoss
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

# Only one top-level worker_init_fn
def worker_init_fn(worker_id):
    import random
    base_seed = getattr(worker_init_fn, 'base_seed', 1234)
    random.seed(base_seed + worker_id)

def trainer_synapse(args, model, snapshot_path, train_dataset=None):
    import random
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    if train_dataset is not None:
        # Balanced patch sampling: ensure each batch has at least one patch with rare classes (1, 4, 5)
        val_ratio = 0.2
        val_size = int(len(train_dataset) * val_ratio)
        train_size = len(train_dataset) - val_size
        db_train, db_val = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
        # Build balanced sampler using db_train
        rare_classes = [1, 4, 5]
        rare_indices = []
        common_indices = []
        for i in range(len(db_train)):
            label = db_train[i]['label'].numpy()
            # For patch-based mode, rare class if any rare pixel in patch
            if any([(label == rc).any() for rc in rare_classes]):
                rare_indices.append(i)
            else:
                common_indices.append(i)
        from torch.utils.data import Sampler
        class BalancedBatchSampler(Sampler):
            def __init__(self, rare_indices, common_indices, batch_size):
                self.rare_indices = rare_indices
                self.common_indices = common_indices
                self.batch_size = batch_size
                self.num_batches = int(np.ceil((len(rare_indices) + len(common_indices)) / batch_size))
            def __iter__(self):
                rare = self.rare_indices.copy()
                common = self.common_indices.copy()
                random.shuffle(rare)
                random.shuffle(common)
                for _ in range(self.num_batches):
                    batch = []
                    if rare:
                        batch.append(rare.pop())
                    while len(batch) < self.batch_size and common:
                        batch.append(common.pop())
                    while len(batch) < self.batch_size and rare:
                        batch.append(rare.pop())
                    yield batch
            def __len__(self):
                return self.num_batches
        sampler = BalancedBatchSampler(rare_indices, common_indices, batch_size)
        worker_init_fn.base_seed = args.seed
        train_loader = DataLoader(db_train, batch_sampler=sampler, num_workers=args.num_workers,
                                  pin_memory=True, worker_init_fn=worker_init_fn)
        val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                                pin_memory=True, worker_init_fn=worker_init_fn)
        # Print class distribution for the first batch
        first_batch = next(iter(train_loader))
        labels = first_batch['label'].cpu().numpy()
        flat = labels.flatten()
        bincount = np.bincount(flat, minlength=args.num_classes)
        print("Class pixel counts in first batch:", bincount)
    else:
        db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                                   transform=transforms.Compose(
                                       [RandomGenerator(output_size=[args.img_size, args.img_size])]))
        db_val = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="val",
                                 transform=transforms.Compose(
                                     [RandomGenerator(output_size=[args.img_size, args.img_size])]))
        print("The length of train set is: {}".format(len(db_train)))
        worker_init_fn.base_seed = args.seed
        train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True, worker_init_fn=worker_init_fn)
        val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                                pin_memory=True, worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    # Compute class weights for UDiadsBibDataset
    if hasattr(train_dataset, 'img_paths') and hasattr(train_dataset, 'mask_paths') and args.dataset.lower() == 'udiads_bib':
        class_counts = np.zeros(args.num_classes, dtype=np.float64)
        print('Computing class weights...')
        for mask_path in train_dataset.mask_paths:
            from PIL import Image
            mask = Image.open(mask_path).convert('RGB')
            mask = np.array(mask)
            # Use the same mapping as in dataset_udiadsbib.py
            mask_class = np.zeros(mask.shape[:2], dtype=np.int64)
            COLOR_MAP = {
                (0, 0, 0): 0,
                (255, 255, 0): 1,
                (0, 255, 255): 2,
                (255, 0, 255): 3,
                (255, 0, 0): 4,
                (0, 255, 0): 5,
            }
            for rgb, cls in COLOR_MAP.items():
                matches = np.all(mask == rgb, axis=-1)
                class_counts[cls] += matches.sum()
        class_freq = class_counts / class_counts.sum()
        weights = 1.0 / (class_freq + 1e-6)
        weights = weights / weights.sum()  # Normalize
        print('Class frequencies:', class_freq)
        print('Class weights:', weights)
        ce_loss = CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).cuda())
    else:
        ce_loss = CrossEntropyLoss()
    from utils import FocalLoss
    dice_loss = DiceLoss(num_classes)
    focal_loss = FocalLoss(gamma=2, weight=torch.tensor(weights, dtype=torch.float32).cuda())
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    writer = SummaryWriter(snapshot_path + '/log')

    best_val_loss = float('inf')
    for epoch in range(args.max_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            # Support both dict and tuple batch
            if isinstance(batch, dict):
                images = batch['image'].cuda()
                labels = batch['label'].cuda()
            else:
                images = batch[0].cuda()
                labels = batch[1].cuda()
            outputs = model(images)
            loss_ce = ce_loss(outputs, labels)
            loss_focal = focal_loss(outputs, labels)
            loss_dice = dice_loss(outputs, labels)
            # Main loss: Weighted sum (CE, Focal, Dice) - more weight to Focal/Dice
            loss = 0.05 * loss_ce + 0.475 * loss_focal + 0.475 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        writer.add_scalar('train/loss', train_loss, epoch)
        logging.info(f"Epoch {epoch+1}/{args.max_epochs} - Train Loss: {train_loss:.4f}")
        print(f"  CrossEntropyLoss: {loss_ce.item():.4f}, FocalLoss: {loss_focal.item():.4f}, DiceLoss: {loss_dice.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    images = batch['image'].cuda()
                    labels = batch['label'].cuda()
                else:
                    images = batch[0].cuda()
                    labels = batch[1].cuda()
                outputs = model(images)
                loss_ce = ce_loss(outputs, labels)
                loss_dice = dice_loss(outputs, labels)
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                val_loss += loss.item()
        val_loss /= len(val_loader)
        writer.add_scalar('val/loss', val_loss, epoch)
        logging.info(f"Epoch {epoch+1}/{args.max_epochs} - Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(snapshot_path, f"best_model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            logging.info(f"Best model saved at epoch {epoch+1} with val loss {val_loss:.4f}")

        # Step the learning rate scheduler
        scheduler.step()
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

    writer.close()
    logging.info("Training completed.")


