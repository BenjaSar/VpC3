"""
Módulo de cálculo avanzado de áreas de habitaciones
Incluye calibración automática y corrección de perspectiva
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from shapely.geometry import Polygon
from shapely.ops import unary_union
import torch


class RoomAreaCalculator:
    """
    Calculador avanzado de áreas de habitaciones

    Características:
    - Calibración automática basada en dimensiones conocidas
    - Corrección de perspectiva
    - Detección de superposiciones
    - Conversión píxel a metros cuadrados
    """

    def __init__(
        self,
        pixel_to_meter: float = 0.02,
        auto_calibrate: bool = True,
        reference_objects: Optional[Dict] = None,
    ):
        """
        Args:
            pixel_to_meter: Factor de conversión inicial
            auto_calibrate: Activar calibración automática
            reference_objects: Objetos de referencia para calibración
                               {clase: dimensión_real_metros}
        """
        self.pixel_to_meter = pixel_to_meter
        self.auto_calibrate = auto_calibrate
        self.reference_objects = reference_objects or {
            "door": 0.9,  # Puerta estándar: 90cm
            "window": 1.5,  # Ventana: 150cm
        }

    def calculate_area_from_mask(
        self, mask: np.ndarray, calibration_factor: Optional[float] = None
    ) -> float:
        """
        Calcular área desde máscara binaria

        Args:
            mask: Máscara binaria (H, W)
            calibration_factor: Factor de calibración personalizado

        Returns:
            Área en metros cuadrados
        """
        factor = calibration_factor if calibration_factor else self.pixel_to_meter

        # Contar píxeles válidos
        area_pixels = np.sum(mask > 0)

        # Convertir a m²
        area_m2 = area_pixels * (factor**2)

        return area_m2

    def calculate_area_from_polygon(
        self,
        polygon: List[Tuple[float, float]],
        calibration_factor: Optional[float] = None,
    ) -> float:
        """
        Calcular área desde polígono

        Args:
            polygon: Lista de puntos [(x1, y1), (x2, y2), ...]
            calibration_factor: Factor de calibración

        Returns:
            Área en metros cuadrados
        """
        factor = calibration_factor if calibration_factor else self.pixel_to_meter

        # Crear polígono de Shapely
        poly = Polygon(polygon)

        # Área en píxeles²
        area_pixels = poly.area

        # Convertir a m²
        area_m2 = area_pixels * (factor**2)

        return area_m2

    def calculate_areas_batch(
        self,
        masks: torch.Tensor,
        labels: torch.Tensor,
        calibration_factor: Optional[float] = None,
    ) -> np.ndarray:
        """
        Calcular áreas para batch de máscaras

        Args:
            masks: Tensor de máscaras (N, H, W)
            labels: Tensor de etiquetas (N,)
            calibration_factor: Factor de calibración

        Returns:
            Array de áreas en m² (N,)
        """
        factor = calibration_factor if calibration_factor else self.pixel_to_meter

        masks_np = masks.cpu().numpy()
        areas = []

        for mask in masks_np:
            area_pixels = np.sum(mask > 0)
            area_m2 = area_pixels * (factor**2)
            areas.append(area_m2)

        return np.array(areas)

    def auto_calibrate_from_reference(
        self, detections: Dict[str, List], reference_class: str = "door"
    ) -> float:
        """
        Calibración automática usando objetos de referencia

        Args:
            detections: Diccionario con detecciones {clase: [bboxes]}
            reference_class: Clase de referencia para calibración

        Returns:
            Nuevo factor de calibración pixel_to_meter
        """
        if reference_class not in detections:
            return self.pixel_to_meter

        if reference_class not in self.reference_objects:
            return self.pixel_to_meter

        # Obtener dimensión real del objeto de referencia
        real_dimension = self.reference_objects[reference_class]

        # Calcular dimensión promedio en píxeles
        bboxes = detections[reference_class]
        if len(bboxes) == 0:
            return self.pixel_to_meter

        widths = [bbox[2] for bbox in bboxes]  # bbox: [x, y, w, h]
        avg_width_pixels = np.mean(widths)

        # Calcular nuevo factor
        new_factor = real_dimension / avg_width_pixels

        return new_factor

    def detect_overlaps(
        self, masks: np.ndarray, labels: np.ndarray
    ) -> List[Tuple[int, int, float]]:
        """
        Detectar superposiciones entre habitaciones

        Args:
            masks: Máscaras de habitaciones (N, H, W)
            labels: Etiquetas de clase (N,)

        Returns:
            Lista de superposiciones: [(idx1, idx2, overlap_area), ...]
        """
        overlaps = []

        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                # Intersección
                intersection = np.logical_and(masks[i], masks[j])
                overlap_area = np.sum(intersection)

                if overlap_area > 0:
                    overlap_m2 = overlap_area * (self.pixel_to_meter**2)
                    overlaps.append((i, j, overlap_m2))

        return overlaps

    def correct_overlaps(
        self,
        masks: np.ndarray,
        labels: np.ndarray,
        priority_order: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Corregir superposiciones usando prioridades

        Args:
            masks: Máscaras originales (N, H, W)
            labels: Etiquetas de clase (N,)
            priority_order: Orden de prioridad de clases

        Returns:
            Máscaras corregidas (N, H, W)
        """
        if priority_order is None:
            # Prioridad por defecto: habitaciones principales > secundarias
            priority_order = [
                "Living Room",
                "Kitchen",
                "Bedroom",
                "Bathroom",
                "Dining Room",
                "Office",
                "Corridor",
                "Storage",
            ]

        corrected_masks = masks.copy()

        # Detectar superposiciones
        overlaps = self.detect_overlaps(masks, labels)

        for i, j, overlap_area in overlaps:
            # Determinar prioridad
            label_i = labels[i]
            label_j = labels[j]

            # Máscara con mayor prioridad mantiene los píxeles
            if label_i in priority_order and label_j in priority_order:
                idx_i = priority_order.index(label_i)
                idx_j = priority_order.index(label_j)

                if idx_i < idx_j:
                    # i tiene mayor prioridad
                    corrected_masks[j] = np.where(
                        corrected_masks[i] > 0, 0, corrected_masks[j]
                    )
                else:
                    # j tiene mayor prioridad
                    corrected_masks[i] = np.where(
                        corrected_masks[j] > 0, 0, corrected_masks[i]
                    )

        return corrected_masks

    def get_room_statistics(
        self, masks: np.ndarray, labels: np.ndarray, class_names: List[str]
    ) -> Dict:
        """
        Calcular estadísticas de habitaciones

        Args:
            masks: Máscaras de habitaciones (N, H, W)
            labels: Etiquetas de clase (N,)
            class_names: Nombres de clases

        Returns:
            Diccionario con estadísticas
        """
        areas = self.calculate_areas_batch(
            torch.from_numpy(masks), torch.from_numpy(labels)
        )

        stats = {
            "total_rooms": len(masks),
            "total_area_m2": np.sum(areas),
            "average_area_m2": np.mean(areas),
            "min_area_m2": np.min(areas) if len(areas) > 0 else 0,
            "max_area_m2": np.max(areas) if len(areas) > 0 else 0,
            "rooms_by_type": {},
            "areas_by_type": {},
        }

        # Agrupar por tipo
        for label, area in zip(labels, areas):
            room_type = class_names[label]

            if room_type not in stats["rooms_by_type"]:
                stats["rooms_by_type"][room_type] = 0
                stats["areas_by_type"][room_type] = []

            stats["rooms_by_type"][room_type] += 1
            stats["areas_by_type"][room_type].append(area)

        # Calcular promedios por tipo
        for room_type in stats["areas_by_type"]:
            areas_list = stats["areas_by_type"][room_type]
            stats["areas_by_type"][room_type] = {
                "count": len(areas_list),
                "total_area": np.sum(areas_list),
                "average_area": np.mean(areas_list),
                "areas": areas_list,
            }

        return stats

    def estimate_scale_from_total_area(
        self, masks: np.ndarray, known_total_area: float
    ) -> float:
        """
        Estimar escala desde área total conocida

        Args:
            masks: Máscaras de todas las habitaciones (N, H, W)
            known_total_area: Área total conocida en m²

        Returns:
            Factor de calibración estimado
        """
        # Área total en píxeles
        total_pixels = np.sum([np.sum(mask > 0) for mask in masks])

        # Calcular factor
        pixel_to_meter = np.sqrt(known_total_area / total_pixels)

        return pixel_to_meter

    def refine_mask_boundaries(
        self, mask: np.ndarray, kernel_size: int = 3
    ) -> np.ndarray:
        """
        Refinar bordes de máscara

        Args:
            mask: Máscara binaria (H, W)
            kernel_size: Tamaño de kernel morfológico

        Returns:
            Máscara refinada
        """
        # Operaciones morfológicas para suavizar bordes
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Cerrar pequeños huecos
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Abrir para eliminar ruido
        mask_refined = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

        return mask_refined


def calculate_perimeter(mask: np.ndarray, pixel_to_meter: float = 0.02) -> float:
    """
    Calcular perímetro de habitación

    Args:
        mask: Máscara binaria
        pixel_to_meter: Factor de conversión

    Returns:
        Perímetro en metros
    """
    # Encontrar contornos
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return 0.0

    # Perímetro del contorno más grande
    perimeter_pixels = cv2.arcLength(contours[0], True)
    perimeter_meters = perimeter_pixels * pixel_to_meter

    return perimeter_meters


if __name__ == "__main__":
    # Test del calculador
    calculator = RoomAreaCalculator(pixel_to_meter=0.02)

    # Crear máscara de prueba (habitación de 100x100 píxeles)
    test_mask = np.zeros((512, 512), dtype=np.uint8)
    test_mask[100:200, 100:200] = 1  # 100x100 = 10,000 píxeles

    # Calcular área
    area = calculator.calculate_area_from_mask(test_mask)
    print(f"Área de prueba: {area:.2f} m²")  # 10,000 * 0.02² = 4 m²

    # Calcular perímetro
    perimeter = calculate_perimeter(test_mask)
    print(f"Perímetro de prueba: {perimeter:.2f} m")  # 400 * 0.02 = 8 m
