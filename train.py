import argparse
import logging
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

dir_img = Path('G:/User/lz/Desktop/yan/1000组sentinel-2标签/images')
dir_mask = Path('G:/User/lz/Desktop/yan/1000组sentinel-2标签/masks')
dir_checkpoint = Path('./checkpoints/')
# Fixed CSV output path
CSV_OUTPUT_PATH = 'metrics.csv'

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0
):
    # Prepare CSV logging
    csv_file = open(CSV_OUTPUT_PATH, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'val_dice', 'val_iou', 'val_recall', 'val_precision', 'val_f1'])

    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. Setup optimizer, scheduler, scaler, criterion
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    best_val_score = -float('inf')
    best_epoch = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type, enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                epoch_loss += loss.item()
                pbar.update(images.shape[0])
                pbar.set_postfix(**{'loss': loss.item()})

        # Validation
        val_metrics = evaluate(model, val_loader, device, amp)
        val_score = val_metrics['dice']
        scheduler.step(val_score)

        # Save best
        if val_score > best_val_score and save_checkpoint:
            best_val_score = val_score
            best_epoch = epoch
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state = model.state_dict()
            state['mask_values'] = dataset.mask_values
            torch.save(state, dir_checkpoint / 'best_model.pth')
            logging.info(f'Best model saved at epoch {epoch}')

        # Save checkpoint every epoch
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state = model.state_dict()
            state['mask_values'] = dataset.mask_values
            torch.save(state, dir_checkpoint / f'epoch_{epoch}.pth')

        # Write to CSV
        csv_writer.writerow([
            epoch,
            epoch_loss / n_train,
            val_metrics['dice'],
            val_metrics['iou'],
            val_metrics['recall'],
            val_metrics['precision'],
            val_metrics['f1']
        ])
        csv_file.flush()
        logging.info(f"Epoch {epoch}: train_loss={epoch_loss / n_train:.4f}, "
                     f"val_dice={val_metrics['dice']:.4f}, val_iou={val_metrics['iou']:.4f}, "
                     f"val_recall={val_metrics['recall']:.4f}, val_precision={val_metrics['precision']:.4f}, "
                     f"val_f1={val_metrics['f1']:.4f}")

    csv_file.close()
    logging.info(f"Training completed. Best Dice: {best_val_score:.4f} at epoch {best_epoch}")
    return best_val_score, best_epoch


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', dest='lr', type=float, default=6e-5, help='Learning rate')
    parser.add_argument('--load', '-f', type=str, default=None, help='Load model path')
    parser.add_argument('--scale', '-s', dest='scale', type=float, default=0.5, help='Downscaling factor')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=30.0, help='Validation percent')
    parser.add_argument('--use_kan', action='store_true', help='Use KAN MLP')
    parser.add_argument('--kan_dropout', type=float, default=0.1, help='KAN dropout')
    parser.add_argument('--kan_type', choices=['vanilla', 'fast'], default='vanilla', help='KAN type')
    parser.add_argument('--amp', action='store_true', help='Mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=True, help='Bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear,
                 use_kan=args.use_kan, kan_type=args.kan_type, dropout=args.kan_dropout)
    model.to(memory_format=torch.channels_last)

    if args.load:
        state = torch.load(args.load, map_location=device)
        state.pop('mask_values', None)
        model.load_state_dict(state)
        logging.info(f'Model loaded from {args.load}')

    model.to(device)
    best_score, best_epoch = train_model(
        model=model,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        img_scale=args.scale,
        val_percent=args.val / 100,
        amp=args.amp
    )
    logging.info(f"Training finished. Best validation Dice: {best_score:.4f} at epoch {best_epoch}")
