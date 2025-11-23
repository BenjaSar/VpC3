"""
Script de inferencia con visualización avanzada
Detecta habitaciones, calcula áreas y genera reportes visuales
"""

import os
import sys
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Imports locales
sys.path.append(str(Path(__file__).parent))
from models.swin_maskrcnn import SwinMaskRCNN
from utils.area_calculator import RoomAreaCalculator, calculate_perimeter
from utils.visualization import FloorPlanVisualizer
from utils.dataset import CubiCasaDataset


class RoomDetector:
    """
    Detector de habitaciones con cálculo de áreas
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        class_names: List[str],
        score_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        pixel_to_meter: float = 0.02,
    ):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        self.area_calculator = RoomAreaCalculator(pixel_to_meter=pixel_to_meter)
        self.visualizer = FloorPlanVisualizer(
            class_names=class_names, score_threshold=score_threshold
        )

    def preprocess_image(
        self, image: np.ndarray, target_size: Tuple[int, int] = (512, 512)
    ) -> Tuple[torch.Tensor, float]:
        """
        Preprocesar imagen para el modelo

        Args:
            image: Imagen RGB (H, W, 3)
            target_size: Tamaño objetivo

        Returns:
            (tensor_imagen, escala)
        """
        original_h, original_w = image.shape[:2]

        # Resize manteniendo aspect ratio
        scale = min(target_size[0] / original_h, target_size[1] / original_w)
        new_h, new_w = int(original_h * scale), int(original_w * scale)

        resized = cv2.resize(image, (new_w, new_h))

        # Padding
        padded = np.zeros((*target_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        # Normalizar
        normalized = padded.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std

        # To tensor (C, H, W)
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)

        return tensor, scale

    def postprocess_outputs(
        self, outputs: Dict, image_shape: Tuple[int, int], scale: float
    ) -> Dict:
        """
        Post-procesar salidas del modelo

        Args:
            outputs: Salidas del modelo
            image_shape: Forma de imagen original
            scale: Factor de escala aplicado

        Returns:
            Predicciones procesadas
        """
        # Simplificación: tomar primer nivel de FPN
        cls_scores = outputs["cls_scores"][0][0]  # (num_classes, H, W)
        bbox_preds = outputs["bbox_preds"][0][0]  # (4, H, W)
        masks = outputs["masks"][0]  # (num_classes, H, W)

        # Obtener predicciones
        scores, labels = torch.max(F.softmax(cls_scores, dim=0), dim=0)

        # Filtrar por threshold
        valid_mask = scores > self.score_threshold

        # Extraer detecciones
        detections = {"boxes": [], "labels": [], "scores": [], "masks": [], "areas": []}

        # Procesar cada detección válida
        h, w = cls_scores.shape[1:]
        for y in range(0, h, 16):  # Submuestreo para eficiencia
            for x in range(0, w, 16):
                if valid_mask[y, x]:
                    label = labels[y, x].item()
                    score = scores[y, x].item()

                    # Bbox simplificado (centrado en punto)
                    bbox_size = 32
                    x1 = max(0, x - bbox_size // 2)
                    y1 = max(0, y - bbox_size // 2)
                    x2 = min(w, x + bbox_size // 2)
                    y2 = min(h, y + bbox_size // 2)

                    # Escalar al tamaño original
                    x1, y1, x2, y2 = [int(c / scale) for c in [x1, y1, x2, y2]]

                    detections["boxes"].append([x1, y1, x2, y2])
                    detections["labels"].append(label)
                    detections["scores"].append(score)

                    # Máscara
                    mask = masks[label].cpu().numpy()
                    mask_resized = cv2.resize(mask, (image_shape[1], image_shape[0]))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    detections["masks"].append(mask_binary)

                    # Calcular área
                    area = self.area_calculator.calculate_area_from_mask(mask_binary)
                    detections["areas"].append(area)

        # Convertir a arrays
        for key in detections:
            if len(detections[key]) > 0:
                detections[key] = np.array(detections[key])
            else:
                detections[key] = np.array([])

        return detections

    def apply_nms(self, detections: Dict) -> Dict:
        """
        Aplicar Non-Maximum Suppression

        Args:
            detections: Detecciones

        Returns:
            Detecciones filtradas
        """
        if len(detections["boxes"]) == 0:
            return detections

        # Simple NMS basado en IoU
        boxes = detections["boxes"]
        scores = detections["scores"]

        # Ordenar por score
        sorted_indices = np.argsort(scores)[::-1]

        keep = []
        while len(sorted_indices) > 0:
            # Mantener el de mayor score
            current = sorted_indices[0]
            keep.append(current)

            if len(sorted_indices) == 1:
                break

            # Calcular IoU con el resto
            current_box = boxes[current]
            other_boxes = boxes[sorted_indices[1:]]

            ious = self._compute_iou_batch(current_box, other_boxes)

            # Filtrar los que tienen IoU bajo
            sorted_indices = sorted_indices[1:][ious < self.nms_threshold]

        # Filtrar detecciones
        for key in detections:
            if len(detections[key]) > 0:
                detections[key] = detections[key][keep]

        return detections

    def _compute_iou_batch(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Calcular IoU entre una box y múltiples boxes"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union_area = box_area + boxes_area - inter_area

        return inter_area / (union_area + 1e-6)

    @torch.no_grad()
    def detect(self, image: np.ndarray) -> Dict:
        """
        Detectar habitaciones en imagen

        Args:
            image: Imagen RGB (H, W, 3)

        Returns:
            Detecciones con boxes, labels, scores, máscaras y áreas
        """
        self.model.eval()

        original_shape = image.shape[:2]

        # Preprocesar
        image_tensor, scale = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)

        # Inferencia
        outputs = self.model(image_tensor)

        # Postprocesar
        detections = self.postprocess_outputs(outputs, original_shape, scale)

        # Aplicar NMS
        detections = self.apply_nms(detections)

        return detections

    def analyze_image(
        self, image_path: str, save_path: Optional[str] = None, show: bool = True
    ) -> Dict:
        """
        Análisis completo de imagen

        Args:
            image_path: Ruta a imagen
            save_path: Ruta para guardar resultados
            show: Mostrar visualización

        Returns:
            Diccionario con resultados y estadísticas
        """
        print(f"\n{'='*70}")
        print(f"{'🏠 ANÁLISIS DE PLANO DE PLANTA':^70}")
        print(f"{'='*70}\n")

        # Cargar imagen
        print(f"📁 Cargando imagen: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"No se pudo cargar: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"   ✓ Imagen cargada: {image.shape}")

        # Detectar
        print("\n🔍 Detectando habitaciones...")
        detections = self.detect(image)

        num_rooms = len(detections["labels"])
        print(f"   ✓ {num_rooms} habitaciones detectadas")

        if num_rooms == 0:
            print("\n⚠️  No se detectaron habitaciones")
            return {"detections": detections}

        # Calcular estadísticas
        print("\n📊 Calculando estadísticas...")
        stats = self.area_calculator.get_room_statistics(
            detections["masks"], detections["labels"], self.class_names
        )

        # Imprimir resultados
        print(f"\n{'Resultados del Análisis':^70}")
        print(f"{'-'*70}")
        print(f"  Total de habitaciones: {stats['total_rooms']}")
        print(f"  Área total: {stats['total_area_m2']:.2f} m²")
        print(f"  Área promedio: {stats['average_area_m2']:.2f} m²")
        print(f"\n{'Habitaciones por tipo:':^70}")
        print(f"{'-'*70}")

        for room_type, data in stats["areas_by_type"].items():
            print(
                f"  {room_type:20s}: {data['count']:2d} habitaciones | "
                f"{data['total_area']:.1f} m² total | "
                f"{data['average_area']:.1f} m² promedio"
            )

        # Visualizar
        if show or save_path:
            print("\n🎨 Generando visualización...")

            # Crear reporte completo
            self.visualizer.create_summary_report(
                image,
                detections["boxes"],
                detections["labels"],
                detections["masks"],
                detections["areas"],
                detections["scores"],
                save_path=save_path if save_path else None,
            )

        print(f"\n{'='*70}")
        print(f"{'✅ ANÁLISIS COMPLETADO':^70}")
        print(f"{'='*70}\n")

        return {"detections": detections, "statistics": stats}


def main():
    parser = argparse.ArgumentParser(
        description="Detectar habitaciones y calcular áreas en planos de planta"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Checkpoint del modelo"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Ruta a imagen de plano de planta"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Ruta para guardar resultados"
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.5, help="Umbral de confianza"
    )
    parser.add_argument(
        "--pixel-to-meter",
        type=float,
        default=0.02,
        help="Factor de conversión pixel a metros",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="No mostrar visualización"
    )

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")

    # Cargar modelo
    print(f"📦 Cargando modelo desde: {args.checkpoint}")

    model = SwinMaskRCNN(num_classes=15, img_size=512)

    # Cargar checkpoint
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("   ✓ Checkpoint cargado")
    else:
        print(f"   ⚠️  Checkpoint no encontrado: {args.checkpoint}")
        print("   Usando modelo no entrenado para demostración")

    model = model.to(device)

    # Crear detector
    detector = RoomDetector(
        model=model,
        device=device,
        class_names=CubiCasaDataset.ROOM_CLASSES,
        score_threshold=args.score_threshold,
        pixel_to_meter=args.pixel_to_meter,
    )

    # Analizar imagen
    results = detector.analyze_image(
        image_path=args.image, save_path=args.output, show=not args.no_show
    )

    print("\n💡 Para entrenar el modelo:")
    print("   python train.py --config configs/swin_maskrcnn_cubicasa.py")


if __name__ == "__main__":
    main()
