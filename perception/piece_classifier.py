"""
Chess piece classifier: CNN-based piece type and color classification.

Architecture: MobileNetV3-Small backbone with custom classification head.
13 classes: empty + 6 piece types × 2 colors
Input: 64×64 RGB square crop
Output: class logits (13-dim)

Designed for ≤30ms inference on GPU, ≤200ms on CPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

logger = logging.getLogger(__name__)


class PieceClass(IntEnum):
    """13-class piece classification."""
    EMPTY = 0
    WHITE_PAWN = 1
    WHITE_KNIGHT = 2
    WHITE_BISHOP = 3
    WHITE_ROOK = 4
    WHITE_QUEEN = 5
    WHITE_KING = 6
    BLACK_PAWN = 7
    BLACK_KNIGHT = 8
    BLACK_BISHOP = 9
    BLACK_ROOK = 10
    BLACK_QUEEN = 11
    BLACK_KING = 12


CLASS_NAMES = [c.name for c in PieceClass]
NUM_CLASSES = len(PieceClass)


@dataclass
class ClassifierConfig:
    """Configuration for the piece classifier."""
    model_path: str = ""
    backbone: str = "mobilenetv3_small"  # mobilenetv3_small, resnet18, efficientnet_b0
    input_size: int = 64
    num_classes: int = NUM_CLASSES
    temperature: float = 1.0  # For calibrated confidence
    device: str = "cpu"
    tta_enabled: bool = False  # Test-time augmentation
    confidence_threshold: float = 0.7


class PieceClassifierNet(nn.Module):
    """
    CNN classifier for chess piece identification.
    
    Uses a pretrained backbone (MobileNetV3-Small by default) with a
    custom classification head. The backbone is fine-tuned end-to-end.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, backbone: str = "mobilenetv3_small"):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone

        if backbone == "mobilenetv3_small":
            base = models.mobilenet_v3_small(weights=None)
            in_features = base.classifier[-1].in_features
            base.classifier[-1] = nn.Identity()
            self.backbone = base
        elif backbone == "resnet18":
            base = models.resnet18(weights=None)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits / self.temperature

    def calibrate_temperature(self, val_logits: torch.Tensor, val_labels: torch.Tensor) -> float:
        """Find optimal temperature using validation set (post-hoc calibration)."""
        best_temp = 1.0
        best_ece = float("inf")
        for t in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
            probs = F.softmax(val_logits / t, dim=1)
            ece = self._compute_ece(probs, val_labels)
            if ece < best_ece:
                best_ece = ece
                best_temp = t
        self.temperature.fill_(best_temp)
        logger.info(f"Calibrated temperature: {best_temp:.2f} (ECE: {best_ece:.4f})")
        return best_temp

    @staticmethod
    def _compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
        """Compute Expected Calibration Error."""
        confidences, predictions = probs.max(dim=1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1)
        for b in range(n_bins):
            lo = b / n_bins
            hi = (b + 1) / n_bins
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() > 0:
                acc = accuracies[mask].float().mean()
                conf = confidences[mask].mean()
                ece += mask.sum().float() / len(labels) * (acc - conf).abs()
        return ece.item()


class PieceClassifier:
    """
    High-level classifier interface for the perception pipeline.
    
    Usage:
        classifier = PieceClassifier(config)
        classifier.load_model("models/piece_classifier.pt")
        result = classifier.classify(square_image)
    """

    def __init__(self, config: ClassifierConfig | None = None):
        self.config = config or ClassifierConfig()
        self.model: Optional[PieceClassifierNet] = None
        self.device = torch.device(self.config.device)
        self._loaded = False

    def load_model(self, path: str | None = None) -> bool:
        """Load a trained model checkpoint."""
        path = path or self.config.model_path
        if not path:
            logger.warning("No model path specified")
            return False
        try:
            self.model = PieceClassifierNet(
                num_classes=self.config.num_classes,
                backbone=self.config.backbone,
            )
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info(f"Loaded classifier from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def classify(self, square_image: np.ndarray) -> tuple[PieceClass, float]:
        """
        Classify a single square image.
        
        Args:
            square_image: HxWx3 uint8 BGR image of a single square.
            
        Returns:
            (predicted_class, confidence)
        """
        if not self._loaded or self.model is None:
            return PieceClass.EMPTY, 0.0

        tensor = self._preprocess(square_image)
        
        with torch.no_grad():
            if self.config.tta_enabled:
                logits = self._tta_forward(tensor)
            else:
                logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

        return PieceClass(pred.item()), conf.item()

    def classify_board(self, warped_image: np.ndarray) -> dict[str, tuple[PieceClass, float]]:
        """Classify all 64 squares from a warped board image."""
        results = {}
        h, w = warped_image.shape[:2]
        sq_h, sq_w = h // 8, w // 8

        for rank in range(8):
            for file in range(8):
                y1 = (7 - rank) * sq_h
                x1 = file * sq_w
                crop = warped_image[y1:y1+sq_h, x1:x1+sq_w]
                sq_name = f"{chr(ord('a')+file)}{rank+1}"
                results[sq_name] = self.classify(crop)

        return results

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a square image for the model."""
        import cv2
        resized = cv2.resize(image, (self.config.input_size, self.config.input_size))
        # BGR to RGB, HWC to CHW, normalize to [0, 1]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0).to(self.device)

    def _tta_forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Test-time augmentation: average predictions over 5 augmented views."""
        logits_list = [self.model(tensor)]
        # Horizontal flip
        logits_list.append(self.model(torch.flip(tensor, [3])))
        # Small rotations via affine
        for angle in [-5, 5]:
            rad = angle * 3.14159 / 180
            cos_a, sin_a = torch.cos(torch.tensor(rad)), torch.sin(torch.tensor(rad))
            theta = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]]).unsqueeze(0).to(self.device)
            grid = F.affine_grid(theta, tensor.size(), align_corners=False)
            rotated = F.grid_sample(tensor, grid, align_corners=False)
            logits_list.append(self.model(rotated))
        # Brightness jitter
        logits_list.append(self.model(tensor * 1.1))
        return torch.stack(logits_list).mean(dim=0)
