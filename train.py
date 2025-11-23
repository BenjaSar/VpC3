"""
Script de entrenamiento con validación avanzada
Incluye mixed precision, gradient accumulation, y métricas completas
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# Imports locales
sys.path.append(str(Path(__file__).parent))
from models.swin_maskrcnn import SwinMaskRCNN
from utils.dataset import CubiCasaDataset, collate_fn
from utils.metrics import DetectionMetrics, AreaEstimationMetrics
from utils.area_calculator import RoomAreaCalculator
from utils.visualization import FloorPlanVisualizer


class Trainer:
    """
    Trainer avanzado con:
    - Mixed Precision Training
    - Gradient Accumulation
    - Learning Rate Scheduling
    - Checkpoint Management
    - WandB Integration (opcional)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device,
        config: dict,
        checkpoint_dir: str = "./checkpoints",
        use_wandb: bool = False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision
        self.use_amp = config.get("use_amp", True)
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient accumulation
        self.accum_steps = config.get("gradient_accumulation_steps", 1)

        # Métricas
        self.metrics = DetectionMetrics(
            num_classes=config["num_classes"],
            class_names=config.get("class_names", None),
        )
        self.area_metrics = AreaEstimationMetrics()
        self.area_calculator = RoomAreaCalculator()

        # Tracking
        self.best_map = 0.0
        self.epoch = 0
        self.global_step = 0

        # WandB (opcional)
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb

                self.wandb = wandb
                wandb.init(project="cubicasa5k-room-detection", config=config)
            except ImportError:
                print("⚠️  WandB not installed. Skipping experiment tracking.")
                self.use_wandb = False

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, verbose=True
        )

    def compute_loss(self, outputs: dict, targets: list) -> dict:
        """
        Calcular pérdidas de detección y segmentación

        Args:
            outputs: Salidas del modelo
            targets: Ground truth

        Returns:
            Diccionario con pérdidas
        """
        losses = {}

        # Loss de clasificación
        cls_loss = 0
        for cls_score in outputs["cls_scores"]:
            # Placeholder - implementar con ground truth
            cls_loss += F.cross_entropy(
                cls_score.flatten(0, -2),
                torch.zeros(cls_score.shape[0], dtype=torch.long, device=self.device),
                reduction="mean",
            )
        losses["cls_loss"] = cls_loss / len(outputs["cls_scores"])

        # Loss de regresión (bbox)
        reg_loss = 0
        for bbox_pred in outputs["bbox_preds"]:
            # Placeholder - implementar con ground truth
            reg_loss += F.smooth_l1_loss(
                bbox_pred, torch.zeros_like(bbox_pred), reduction="mean"
            )
        losses["reg_loss"] = reg_loss / len(outputs["bbox_preds"])

        # Loss de máscara
        mask_loss = F.binary_cross_entropy_with_logits(
            outputs["masks"], torch.zeros_like(outputs["masks"]), reduction="mean"
        )
        losses["mask_loss"] = mask_loss

        # Loss total
        losses["total_loss"] = (
            self.config.get("cls_weight", 1.0) * losses["cls_loss"]
            + self.config.get("reg_weight", 1.0) * losses["reg_loss"]
            + self.config.get("mask_weight", 1.0) * losses["mask_loss"]
        )

        return losses

    def train_epoch(self) -> dict:
        """
        Entrenar una época

        Returns:
            Métricas de entrenamiento
        """
        self.model.train()
        epoch_losses = {
            "total_loss": 0.0,
            "cls_loss": 0.0,
            "reg_loss": 0.0,
            "mask_loss": 0.0,
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1} [Train]")

        for batch_idx, batch in enumerate(pbar):
            images = batch["images"].to(self.device)
            # targets = ... (procesar ground truth)

            # Forward pass con mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                losses = self.compute_loss(outputs, None)  # targets aquí
                loss = losses["total_loss"] / self.accum_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.accum_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Actualizar métricas
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()

            # Actualizar barra de progreso
            pbar.set_postfix(
                {
                    "loss": loss.item() * self.accum_steps,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )

        # Promediar pérdidas
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> dict:
        """
        Validar modelo

        Returns:
            Métricas de validación
        """
        self.model.eval()
        self.metrics.reset()

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch + 1} [Val]")

        for batch in pbar:
            images = batch["images"].to(self.device)

            # Forward pass
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)

            # Procesar predicciones
            # pred_boxes, pred_labels, pred_scores = postprocess(outputs)
            # self.metrics.update(...)

        # Calcular métricas
        val_metrics = self.metrics.evaluate()

        return val_metrics

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        Guardar checkpoint

        Args:
            filename: Nombre del archivo
            is_best: Si es el mejor modelo hasta ahora
        """
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_map": self.best_map,
            "config": self.config,
        }

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Guardar checkpoint
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

        # Guardar como mejor modelo
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved: mAP = {self.best_map:.4f}")

    def train(self, num_epochs: int):
        """
        Loop de entrenamiento completo

        Args:
            num_epochs: Número de épocas
        """
        print("\n" + "=" * 70)
        print(f"{'🚀 COMENZANDO ENTRENAMIENTO':^70}")
        print("=" * 70)
        print(f"\nConfiguración:")
        print(f"  - Device: {self.device}")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - Batch size: {self.config['batch_size']}")
        print(f"  - Learning rate: {self.config['learning_rate']}")
        print(f"  - Mixed precision: {self.use_amp}")
        print(f"  - Gradient accumulation: {self.accum_steps}")
        print("=" * 70 + "\n")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Entrenar
            train_metrics = self.train_epoch()

            # Validar
            val_metrics = self.validate()

            # Actualizar scheduler
            self.scheduler.step(val_metrics["mAP"])

            # Imprimir métricas
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"  Val mAP: {val_metrics['mAP']:.4f}")
            print(f"  Val Precision: {val_metrics['mean_precision']:.4f}")
            print(f"  Val Recall: {val_metrics['mean_recall']:.4f}")

            # Log a WandB
            if self.use_wandb:
                self.wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_metrics["total_loss"],
                        "val_map": val_metrics["mAP"],
                        "val_precision": val_metrics["mean_precision"],
                        "val_recall": val_metrics["mean_recall"],
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                )

            # Guardar checkpoint
            is_best = val_metrics["mAP"] > self.best_map
            if is_best:
                self.best_map = val_metrics["mAP"]

            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth", is_best=is_best)

        print("\n" + "=" * 70)
        print(f"{'✅ ENTRENAMIENTO COMPLETADO':^70}")
        print("=" * 70)
        print(f"\nMejor mAP: {self.best_map:.4f}")
        print(f"Checkpoints guardados en: {self.checkpoint_dir}")
        print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Entrenar Swin Mask R-CNN")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/swin_maskrcnn_cubicasa.py",
        help="Archivo de configuración",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/cubicasa5k",
        help="Directorio del dataset",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=12, help="Número de épocas")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directorio para checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint para continuar entrenamiento",
    )
    parser.add_argument("--wandb", action="store_true", help="Usar Weights & Biases")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # Configuración
    config = {
        "num_classes": 15,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "use_amp": True,
        "gradient_accumulation_steps": 4,  # Para optimizar 4GB VRAM
        "cls_weight": 1.0,
        "reg_weight": 1.0,
        "mask_weight": 1.0,
        "class_names": CubiCasaDataset.ROOM_CLASSES,
    }

    # Modelo
    print("\n🏗️  Construyendo modelo...")
    model = SwinMaskRCNN(num_classes=config["num_classes"], img_size=512)
    model = model.to(device)

    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total de parámetros: {total_params:,}")

    # Dataset (nota: necesitarías implementar el loader real)
    print("\n📚 Cargando dataset...")
    print("   ⚠️  NOTA: Asegúrate de haber descargado el dataset con:")
    print("      python download_dataset.py")

    # Placeholder datasets
    # train_dataset = CubiCasaDataset(...)
    # val_dataset = CubiCasaDataset(...)

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config['batch_size'],
    #     shuffle=True,
    #     num_workers=2,
    #     collate_fn=collate_fn
    # )

    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=config['batch_size'],
    #     shuffle=False,
    #     num_workers=2,
    #     collate_fn=collate_fn
    # )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=0.05
    )

    # Trainer
    # trainer = Trainer(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     optimizer=optimizer,
    #     device=device,
    #     config=config,
    #     checkpoint_dir=args.checkpoint_dir,
    #     use_wandb=args.wandb
    # )

    # # Entrenar
    # trainer.train(num_epochs=args.epochs)

    print("\n⚠️  Para ejecutar el entrenamiento completo:")
    print("   1. Descarga el dataset: python download_dataset.py")
    print("   2. Descomenta las secciones de dataset y trainer")
    print("   3. Ejecuta: python train.py")


if __name__ == "__main__":
    main()
