#!/usr/bin/env python3
"""
Train the chess piece classifier.

Generates synthetic training data (chess square crops with domain randomization)
and trains a MobileNetV3-Small classifier for 13-class piece identification.

Usage:
    python scripts/train_classifier.py --epochs 30 --batch-size 64
    python scripts/train_classifier.py --backbone resnet18 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perception.piece_classifier import (
    CLASS_NAMES,
    NUM_CLASSES,
    PieceClass,
    PieceClassifierNet,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Piece rendering parameters (simplified synthetic rendering)
PIECE_SYMBOLS = {
    PieceClass.WHITE_PAWN: "P", PieceClass.WHITE_KNIGHT: "N",
    PieceClass.WHITE_BISHOP: "B", PieceClass.WHITE_ROOK: "R",
    PieceClass.WHITE_QUEEN: "Q", PieceClass.WHITE_KING: "K",
    PieceClass.BLACK_PAWN: "p", PieceClass.BLACK_KNIGHT: "n",
    PieceClass.BLACK_BISHOP: "b", PieceClass.BLACK_ROOK: "r",
    PieceClass.BLACK_QUEEN: "q", PieceClass.BLACK_KING: "k",
}

PIECE_RADII = {
    PieceClass.WHITE_PAWN: 8, PieceClass.WHITE_KNIGHT: 10,
    PieceClass.WHITE_BISHOP: 10, PieceClass.WHITE_ROOK: 11,
    PieceClass.WHITE_QUEEN: 12, PieceClass.WHITE_KING: 12,
    PieceClass.BLACK_PAWN: 8, PieceClass.BLACK_KNIGHT: 10,
    PieceClass.BLACK_BISHOP: 10, PieceClass.BLACK_ROOK: 11,
    PieceClass.BLACK_QUEEN: 12, PieceClass.BLACK_KING: 12,
}


class SyntheticChessDataset(Dataset):
    """
    Synthetic chess square dataset with domain randomization.
    
    Generates 64x64 crops of chess squares with:
    - Random square color (light/dark with jitter)
    - Random piece placement with shape variation
    - Random lighting/brightness
    - Random noise
    - Random rotation (±5°)
    """

    def __init__(self, num_samples: int = 10000, img_size: int = 64, seed: int = 42):
        self.num_samples = num_samples
        self.img_size = img_size
        self.rng = np.random.RandomState(seed)
        
        # Pre-generate labels with class balance
        self.labels = []
        samples_per_class = num_samples // NUM_CLASSES
        for cls in range(NUM_CLASSES):
            self.labels.extend([cls] * samples_per_class)
        # Fill remainder with random classes
        while len(self.labels) < num_samples:
            self.labels.append(self.rng.randint(0, NUM_CLASSES))
        self.rng.shuffle(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = self._render_square(label, idx)
        
        # To tensor: HWC -> CHW, normalize
        tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor, label

    def _render_square(self, label: int, seed_offset: int) -> np.ndarray:
        """Render a synthetic chess square image."""
        rng = np.random.RandomState(seed_offset * 13 + label * 7)
        s = self.img_size
        
        # Square background color (light or dark with jitter)
        is_light = rng.random() > 0.5
        if is_light:
            base = rng.randint(170, 230)
            bg = np.array([base + rng.randint(-15, 15),
                          base + rng.randint(-10, 20),
                          base + rng.randint(-20, 10)], dtype=np.uint8)
        else:
            base = rng.randint(60, 130)
            bg = np.array([base + rng.randint(-15, 15),
                          base + rng.randint(-10, 10),
                          base + rng.randint(-15, 15)], dtype=np.uint8)
        
        img = np.full((s, s, 3), bg, dtype=np.uint8)
        
        # Add subtle texture
        noise = rng.randint(-8, 8, (s, s, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        if label == PieceClass.EMPTY:
            return self._apply_augmentation(img, rng)
        
        # Draw piece
        piece_class = PieceClass(label)
        cx, cy = s // 2 + rng.randint(-3, 4), s // 2 + rng.randint(-3, 4)
        radius = PIECE_RADII.get(piece_class, 10) + rng.randint(-2, 3)
        
        # Piece color
        if label <= 6:  # White pieces
            piece_color = (rng.randint(200, 255), rng.randint(200, 255), rng.randint(190, 250))
        else:  # Black pieces
            piece_color = (rng.randint(20, 80), rng.randint(20, 80), rng.randint(20, 80))
        
        # Draw piece body (circle + shape variation)
        cv2.circle(img, (cx, cy), radius, piece_color, -1)
        
        # Add piece-specific features
        if piece_class in (PieceClass.WHITE_ROOK, PieceClass.BLACK_ROOK):
            # Rook: square-ish top
            cv2.rectangle(img, (cx-radius, cy-radius), (cx+radius, cy-radius//2), piece_color, -1)
        elif piece_class in (PieceClass.WHITE_BISHOP, PieceClass.BLACK_BISHOP):
            # Bishop: pointed top
            pts = np.array([[cx, cy-radius-3], [cx-5, cy-3], [cx+5, cy-3]])
            cv2.fillPoly(img, [pts], piece_color)
        elif piece_class in (PieceClass.WHITE_QUEEN, PieceClass.BLACK_QUEEN):
            # Queen: larger with crown
            cv2.circle(img, (cx, cy-radius+2), 4, piece_color, -1)
        elif piece_class in (PieceClass.WHITE_KING, PieceClass.BLACK_KING):
            # King: cross on top
            cv2.line(img, (cx, cy-radius-5), (cx, cy-radius+3), piece_color, 2)
            cv2.line(img, (cx-4, cy-radius-2), (cx+4, cy-radius-2), piece_color, 2)
        elif piece_class in (PieceClass.WHITE_KNIGHT, PieceClass.BLACK_KNIGHT):
            # Knight: asymmetric
            pts = np.array([[cx-3, cy+radius], [cx+5, cy-radius], [cx+radius, cy]])
            cv2.fillPoly(img, [pts], piece_color)
        
        # Add edge/shadow
        shadow_color = tuple(max(0, c - 40) for c in piece_color)
        cv2.circle(img, (cx+1, cy+1), radius, shadow_color, 1)
        
        return self._apply_augmentation(img, rng)

    def _apply_augmentation(self, img: np.ndarray, rng) -> np.ndarray:
        """Apply domain randomization augmentations."""
        s = self.img_size
        
        # Brightness jitter
        factor = rng.uniform(0.7, 1.3)
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        
        # Gaussian blur (simulate focus variation)
        if rng.random() > 0.5:
            ksize = rng.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
        # Small rotation
        angle = rng.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((s/2, s/2), angle, 1.0)
        img = cv2.warpAffine(img, M, (s, s), borderMode=cv2.BORDER_REFLECT)
        
        # Gaussian noise
        if rng.random() > 0.3:
            noise = rng.normal(0, rng.uniform(3, 10), img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img


def train(args):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create datasets
    logger.info(f"Generating {args.num_samples} synthetic samples...")
    full_dataset = SyntheticChessDataset(
        num_samples=args.num_samples,
        img_size=args.input_size,
        seed=args.seed,
    )
    
    # Split: 80% train, 10% val, 10% test
    n = len(full_dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    # Create model
    model = PieceClassifierNet(num_classes=NUM_CLASSES, backbone=args.backbone)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {args.backbone}, params: {total_params:,}")
    
    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            train_correct += (logits.argmax(1) == batch_y).sum().item()
            train_total += batch_x.size(0)
        
        scheduler.step()
        
        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                val_correct += (logits.argmax(1) == batch_y).sum().item()
                val_total += batch_x.size(0)
                all_logits.append(logits.cpu())
                all_labels.append(batch_y.cpu())
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        epoch_time = time.time() - t0
        
        history["train_loss"].append(train_loss / train_total)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss / val_total)
        history["val_acc"].append(val_acc)
        
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss/train_total:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss/val_total:.4f} Acc: {val_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | {epoch_time:.1f}s"
        )
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "backbone": args.backbone,
                "num_classes": NUM_CLASSES,
                "class_names": CLASS_NAMES,
            }, output_dir / "piece_classifier_best.pt")
            logger.info(f"  ★ New best val acc: {val_acc:.4f}")
    
    # Temperature calibration on validation set
    logger.info("Calibrating temperature...")
    val_logits = torch.cat(all_logits)
    val_labels = torch.cat(all_labels)
    model.calibrate_temperature(val_logits, val_labels)
    
    # Final test evaluation
    logger.info("Evaluating on test set...")
    model.eval()
    test_correct, test_total = 0, 0
    per_class_correct = torch.zeros(NUM_CLASSES)
    per_class_total = torch.zeros(NUM_CLASSES)
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x).argmax(1)
            test_correct += (preds == batch_y).sum().item()
            test_total += batch_x.size(0)
            for c in range(NUM_CLASSES):
                mask = batch_y == c
                per_class_correct[c] += (preds[mask] == c).sum().item()
                per_class_total[c] += mask.sum().item()
    
    test_acc = test_correct / test_total
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RESULTS")
    logger.info(f"Best val acc: {best_val_acc:.4f} (epoch {best_epoch})")
    logger.info(f"Test accuracy: {test_acc:.4f}")
    logger.info(f"Temperature: {model.temperature.item():.2f}")
    
    # Per-class breakdown
    logger.info(f"\nPer-class accuracy:")
    macro_recall = 0.0
    for c in range(NUM_CLASSES):
        if per_class_total[c] > 0:
            acc = per_class_correct[c] / per_class_total[c]
            macro_recall += acc
            logger.info(f"  {CLASS_NAMES[c]:15s}: {acc:.4f} ({int(per_class_correct[c])}/{int(per_class_total[c])})")
    macro_recall /= NUM_CLASSES
    logger.info(f"\nMacro recall: {macro_recall:.4f}")
    
    # Save final model with calibration
    final_path = output_dir / "piece_classifier.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": best_epoch,
        "val_acc": best_val_acc,
        "test_acc": test_acc,
        "macro_recall": macro_recall,
        "temperature": model.temperature.item(),
        "backbone": args.backbone,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "history": history,
    }, final_path)
    logger.info(f"Saved final model to {final_path}")
    
    # Save model card
    card = {
        "model": args.backbone,
        "params": total_params,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "macro_recall": float(macro_recall),
        "temperature": float(model.temperature.item()),
        "num_classes": NUM_CLASSES,
        "input_size": args.input_size,
        "train_samples": n_train,
        "val_samples": n_val,
        "test_samples": n_test,
        "seed": args.seed,
        "device": str(device),
    }
    with open(output_dir / "model_card.json", "w") as f:
        json.dump(card, f, indent=2)
    
    return test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train chess piece classifier")
    parser.add_argument("--backbone", default="mobilenetv3_small", choices=["mobilenetv3_small", "resnet18"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="models/classifier")
    args = parser.parse_args()
    
    train(args)
