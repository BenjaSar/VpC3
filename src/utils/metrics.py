"""
Métricas avanzadas de evaluación
mAP, IoU, precisión de áreas, etc.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class DetectionMetrics:
    """
    Calculador de métricas para detección de objetos y segmentación
    """

    def __init__(
        self,
        num_classes: int,
        iou_thresholds: List[float] = [0.5, 0.75, 0.95],
        class_names: Optional[List[str]] = None,
    ):
        """
        Args:
            num_classes: Número de clases
            iou_thresholds: Umbrales de IoU para mAP
            class_names: Nombres de clases para reporte
        """
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

        # Almacenar resultados
        self.reset()

    def reset(self):
        """Reiniciar métricas"""
        self.predictions = defaultdict(list)
        self.ground_truths = defaultdict(list)

    def update(
        self,
        pred_boxes: np.ndarray,
        pred_labels: np.ndarray,
        pred_scores: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
        image_id: int = 0,
    ):
        """
        Actualizar con predicciones y ground truth

        Args:
            pred_boxes: Predicciones (N, 4) [x1, y1, x2, y2]
            pred_labels: Etiquetas predichas (N,)
            pred_scores: Scores de confianza (N,)
            gt_boxes: Ground truth boxes (M, 4)
            gt_labels: Ground truth labels (M,)
            image_id: ID de imagen
        """
        self.predictions[image_id] = {
            "boxes": pred_boxes,
            "labels": pred_labels,
            "scores": pred_scores,
        }

        self.ground_truths[image_id] = {"boxes": gt_boxes, "labels": gt_labels}

    def compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calcular IoU entre dos boxes

        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]

        Returns:
            IoU score
        """
        # Coordenadas de intersección
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Área de intersección
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Áreas de boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Área de unión
        union_area = box1_area + box2_area - inter_area

        # IoU
        iou = inter_area / union_area if union_area > 0 else 0.0

        return iou

    def compute_iou_batch(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        Calcular IoU entre dos conjuntos de boxes

        Args:
            boxes1: (N, 4)
            boxes2: (M, 4)

        Returns:
            IoU matrix (N, M)
        """
        N = len(boxes1)
        M = len(boxes2)
        ious = np.zeros((N, M))

        for i in range(N):
            for j in range(M):
                ious[i, j] = self.compute_iou(boxes1[i], boxes2[j])

        return ious

    def compute_ap(self, recalls: np.ndarray, precisions: np.ndarray) -> float:
        """
        Calcular Average Precision

        Args:
            recalls: Array de recall values
            precisions: Array de precision values

        Returns:
            AP score
        """
        # Interpolar precision
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        recalls = np.concatenate(([0.0], recalls, [1.0]))

        # Hacer precision monótona decreciente
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        # Calcular AP usando interpolación de 11 puntos
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0

        return ap

    def evaluate(self) -> Dict:
        """
        Calcular todas las métricas

        Returns:
            Diccionario con métricas
        """
        metrics = {}

        # mAP por IoU threshold
        for iou_threshold in self.iou_thresholds:
            aps = []

            for class_id in range(self.num_classes):
                ap = self._compute_class_ap(class_id, iou_threshold)
                aps.append(ap)

            map_score = np.mean(aps)
            metrics[f"mAP@{iou_threshold}"] = map_score
            metrics[f"APs@{iou_threshold}"] = aps

        # mAP promedio (COCO style)
        metrics["mAP"] = np.mean([metrics[f"mAP@{iou}"] for iou in self.iou_thresholds])

        # Precision y Recall por clase
        precision_per_class, recall_per_class = self._compute_precision_recall()
        metrics["precision_per_class"] = precision_per_class
        metrics["recall_per_class"] = recall_per_class

        # Promedios
        metrics["mean_precision"] = np.mean(precision_per_class)
        metrics["mean_recall"] = np.mean(recall_per_class)

        return metrics

    def _compute_class_ap(self, class_id: int, iou_threshold: float) -> float:
        """
        Calcular AP para una clase específica

        Args:
            class_id: ID de clase
            iou_threshold: Umbral de IoU

        Returns:
            AP score
        """
        # Recopilar todas las predicciones y GT de esta clase
        all_pred_boxes = []
        all_pred_scores = []
        all_gt_boxes = []
        all_image_ids = []

        for img_id in self.predictions.keys():
            pred = self.predictions[img_id]
            gt = self.ground_truths.get(
                img_id, {"boxes": np.array([]), "labels": np.array([])}
            )

            # Predicciones de esta clase
            pred_mask = pred["labels"] == class_id
            if np.sum(pred_mask) > 0:
                all_pred_boxes.extend(pred["boxes"][pred_mask])
                all_pred_scores.extend(pred["scores"][pred_mask])
                all_image_ids.extend([img_id] * np.sum(pred_mask))

            # Ground truth de esta clase
            gt_mask = gt["labels"] == class_id
            if np.sum(gt_mask) > 0:
                all_gt_boxes.extend(gt["boxes"][gt_mask])

        if len(all_pred_boxes) == 0 or len(all_gt_boxes) == 0:
            return 0.0

        # Ordenar predicciones por score descendente
        sorted_indices = np.argsort(all_pred_scores)[::-1]
        all_pred_boxes = np.array(all_pred_boxes)[sorted_indices]
        all_pred_scores = np.array(all_pred_scores)[sorted_indices]

        # Calcular TP y FP
        tp = np.zeros(len(all_pred_boxes))
        fp = np.zeros(len(all_pred_boxes))
        matched_gt = set()

        for i, pred_box in enumerate(all_pred_boxes):
            max_iou = 0
            max_gt_idx = -1

            # Encontrar GT con mayor IoU
            for j, gt_box in enumerate(all_gt_boxes):
                if j not in matched_gt:
                    iou = self.compute_iou(pred_box, gt_box)
                    if iou > max_iou:
                        max_iou = iou
                        max_gt_idx = j

            # Determinar TP o FP
            if max_iou >= iou_threshold:
                tp[i] = 1
                matched_gt.add(max_gt_idx)
            else:
                fp[i] = 1

        # Calcular precision y recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / len(all_gt_boxes)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        # Calcular AP
        ap = self.compute_ap(recalls, precisions)

        return ap

    def _compute_precision_recall(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcular precision y recall por clase

        Returns:
            (precision_per_class, recall_per_class)
        """
        precision_per_class = np.zeros(self.num_classes)
        recall_per_class = np.zeros(self.num_classes)

        for class_id in range(self.num_classes):
            tp = 0
            fp = 0
            fn = 0

            for img_id in self.predictions.keys():
                pred = self.predictions[img_id]
                gt = self.ground_truths.get(
                    img_id, {"boxes": np.array([]), "labels": np.array([])}
                )

                # Predicciones y GT de esta clase
                pred_mask = pred["labels"] == class_id
                gt_mask = gt["labels"] == class_id

                num_pred = np.sum(pred_mask)
                num_gt = np.sum(gt_mask)

                # Simplificado: contar correctos por umbral de IoU=0.5
                if num_pred > 0 and num_gt > 0:
                    pred_boxes = pred["boxes"][pred_mask]
                    gt_boxes = gt["boxes"][gt_mask]

                    ious = self.compute_iou_batch(pred_boxes, gt_boxes)
                    matched = np.max(ious, axis=1) >= 0.5

                    tp += np.sum(matched)
                    fp += num_pred - np.sum(matched)
                    fn += num_gt - np.sum(matched)
                elif num_pred > 0:
                    fp += num_pred
                elif num_gt > 0:
                    fn += num_gt

            # Calcular métricas
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            precision_per_class[class_id] = precision
            recall_per_class[class_id] = recall

        return precision_per_class, recall_per_class

    def print_summary(self, metrics: Dict):
        """
        Imprimir resumen de métricas

        Args:
            metrics: Diccionario de métricas
        """
        print("\n" + "=" * 60)
        print("MÉTRICAS DE EVALUACIÓN")
        print("=" * 60)

        print(f"\nmAP (mean Average Precision): {metrics['mAP']:.4f}")

        for iou_threshold in self.iou_thresholds:
            print(f"mAP@{iou_threshold}: {metrics[f'mAP@{iou_threshold}']:.4f}")

        print(f"\nPrecision promedio: {metrics['mean_precision']:.4f}")
        print(f"Recall promedio: {metrics['mean_recall']:.4f}")

        print("\n" + "-" * 60)
        print("Métricas por clase:")
        print("-" * 60)

        for i, class_name in enumerate(self.class_names):
            precision = metrics["precision_per_class"][i]
            recall = metrics["recall_per_class"][i]
            print(
                f"{class_name:20s} | Precision: {precision:.4f} | Recall: {recall:.4f}"
            )

        print("=" * 60 + "\n")


class AreaEstimationMetrics:
    """
    Métricas específicas para estimación de áreas
    """

    def __init__(self):
        self.predictions = []
        self.ground_truths = []

    def update(self, pred_areas: np.ndarray, gt_areas: np.ndarray):
        """
        Actualizar con áreas predichas y ground truth

        Args:
            pred_areas: Áreas predichas (N,)
            gt_areas: Áreas ground truth (N,)
        """
        self.predictions.extend(pred_areas)
        self.ground_truths.extend(gt_areas)

    def evaluate(self) -> Dict:
        """
        Calcular métricas de error de área

        Returns:
            Diccionario con métricas
        """
        pred = np.array(self.predictions)
        gt = np.array(self.ground_truths)

        # Error absoluto
        mae = np.mean(np.abs(pred - gt))

        # Error cuadrático medio
        mse = np.mean((pred - gt) ** 2)
        rmse = np.sqrt(mse)

        # Error porcentual
        mape = np.mean(np.abs((pred - gt) / (gt + 1e-6))) * 100

        # R² score
        ss_res = np.sum((gt - pred) ** 2)
        ss_tot = np.sum((gt - np.mean(gt)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

    def plot_comparison(self, save_path: Optional[str] = None):
        """
        Gráfico de comparación predicción vs ground truth

        Args:
            save_path: Ruta para guardar
        """
        pred = np.array(self.predictions)
        gt = np.array(self.ground_truths)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot
        ax1.scatter(gt, pred, alpha=0.5, s=30)
        ax1.plot([gt.min(), gt.max()], [gt.min(), gt.max()], "r--", lw=2, label="Ideal")
        ax1.set_xlabel("Ground Truth Area (m²)")
        ax1.set_ylabel("Predicted Area (m²)")
        ax1.set_title("Área Predicha vs Ground Truth")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Error distribution
        errors = pred - gt
        ax2.hist(errors, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        ax2.axvline(x=0, color="red", linestyle="--", lw=2, label="Error = 0")
        ax2.set_xlabel("Error (m²)")
        ax2.set_ylabel("Frecuencia")
        ax2.set_title("Distribución de Errores")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()


if __name__ == "__main__":
    # Test de métricas
    metrics = DetectionMetrics(
        num_classes=3, class_names=["Bedroom", "Kitchen", "Living Room"]
    )

    # Datos de prueba
    pred_boxes = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])
    pred_labels = np.array([0, 1])
    pred_scores = np.array([0.9, 0.85])

    gt_boxes = np.array([[12, 12, 52, 52], [58, 58, 98, 98]])
    gt_labels = np.array([0, 1])

    metrics.update(
        pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, image_id=0
    )

    results = metrics.evaluate()
    metrics.print_summary(results)
