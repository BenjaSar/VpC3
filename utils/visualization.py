"""
Visualización avanzada de resultados
Detecciones, máscaras, áreas y métricas
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MPLPolygon
from typing import Dict, List, Tuple, Optional
import seaborn as sns
import torch


# Paleta de colores para clases de habitaciones
ROOM_COLORS = {
    "Bedroom": (255, 99, 71),  # Tomato
    "Kitchen": (255, 165, 0),  # Orange
    "Living Room": (60, 179, 113),  # MediumSeaGreen
    "Bathroom": (100, 149, 237),  # CornflowerBlue
    "Dining Room": (255, 215, 0),  # Gold
    "Corridor": (169, 169, 169),  # DarkGray
    "Balcony": (135, 206, 250),  # LightSkyBlue
    "Storage": (210, 180, 140),  # Tan
    "Office": (147, 112, 219),  # MediumPurple
    "Laundry": (176, 224, 230),  # PowderBlue
    "Garage": (128, 128, 128),  # Gray
    "Terrace": (144, 238, 144),  # LightGreen
    "Closet": (222, 184, 135),  # BurlyWood
    "Entrance": (205, 133, 63),  # Peru
    "Other": (192, 192, 192),  # Silver
}


class FloorPlanVisualizer:
    """
    Visualizador avanzado para planos de planta y detecciones
    """

    def __init__(
        self,
        class_names: List[str],
        score_threshold: float = 0.5,
        colors: Optional[Dict] = None,
    ):
        """
        Args:
            class_names: Lista de nombres de clases
            score_threshold: Umbral de confianza para visualizar
            colors: Diccionario personalizado de colores {clase: (R,G,B)}
        """
        self.class_names = class_names
        self.score_threshold = score_threshold
        self.colors = colors if colors else ROOM_COLORS

        # Generar colores aleatorios para clases sin color definido
        for class_name in class_names:
            if class_name not in self.colors:
                self.colors[class_name] = tuple(np.random.randint(0, 255, 3).tolist())

    def draw_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
        scores: Optional[np.ndarray] = None,
        areas: Optional[np.ndarray] = None,
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Dibujar bounding boxes en imagen

        Args:
            image: Imagen RGB (H, W, 3)
            boxes: Bounding boxes (N, 4) en formato [x1, y1, x2, y2]
            labels: Etiquetas de clase (N,)
            scores: Scores de confianza (N,)
            areas: Áreas en m² (N,)
            thickness: Grosor de líneas

        Returns:
            Imagen con boxes dibujados
        """
        img_draw = image.copy()

        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box.astype(int)

            # Color de la clase
            class_name = self.class_names[label]
            color = self.colors.get(class_name, (255, 255, 255))

            # Dibujar rectángulo
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)

            # Texto de etiqueta
            text_parts = [class_name]

            if scores is not None:
                text_parts.append(f"{scores[i]:.2f}")

            if areas is not None:
                text_parts.append(f"{areas[i]:.1f}m²")

            text = " | ".join(text_parts)

            # Fondo del texto
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                img_draw, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1
            )

            # Texto
            cv2.putText(
                img_draw,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        return img_draw

    def draw_masks(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Dibujar máscaras de segmentación

        Args:
            image: Imagen RGB (H, W, 3)
            masks: Máscaras binarias (N, H, W) o (N, 1, H, W)
            labels: Etiquetas de clase (N,)
            alpha: Transparencia de máscaras

        Returns:
            Imagen con máscaras dibujadas
        """
        img_draw = image.copy().astype(np.float32)
        overlay = img_draw.copy()

        for mask, label in zip(masks, labels):
            # Color de la clase
            class_name = self.class_names[label]
            color = self.colors.get(class_name, (255, 255, 255))

            # Asegurar que mask sea 2D (H, W)
            # Si tiene forma (1, H, W), hacer squeeze
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]  # Ahora (H, W)
            elif mask.ndim == 3:
                # Si tiene múltiples canales, tomar el primero
                mask = mask[0]

            # Aplicar color a la máscara
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = color

        # Mezclar con transparencia
        img_draw = cv2.addWeighted(img_draw, 1 - alpha, overlay, alpha, 0)

        return img_draw.astype(np.uint8)

    def draw_combined(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
        masks: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
        areas: Optional[np.ndarray] = None,
        show_boxes: bool = True,
        show_masks: bool = True,
        mask_alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Visualización completa: boxes + máscaras + áreas

        Args:
            image: Imagen RGB (H, W, 3)
            boxes: Bounding boxes (N, 4)
            labels: Etiquetas (N,)
            masks: Máscaras (N, H, W)
            scores: Scores (N,)
            areas: Áreas en m² (N,)
            show_boxes: Mostrar bounding boxes
            show_masks: Mostrar máscaras
            mask_alpha: Transparencia de máscaras

        Returns:
            Imagen con visualización completa
        """
        img_draw = image.copy()

        # Dibujar máscaras primero (fondo)
        if show_masks and masks is not None:
            img_draw = self.draw_masks(img_draw, masks, labels, alpha=mask_alpha)

        # Dibujar boxes encima
        if show_boxes:
            img_draw = self.draw_boxes(
                img_draw, boxes, labels, scores, areas, thickness=2
            )

        return img_draw

    def plot_detections(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
        masks: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
        areas: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
    ):
        """
        Visualización con matplotlib (mejor calidad)

        Args:
            image: Imagen RGB
            boxes: Bounding boxes (N, 4) en formato [x1, y1, x2, y2]
            labels: Etiquetas (N,)
            masks: Máscaras (N, H, W)
            scores: Scores (N,)
            areas: Áreas (N,)
            save_path: Ruta para guardar imagen
            figsize: Tamaño de figura
        """
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(image)

        # Dibujar máscaras
        if masks is not None:
            for mask, label in zip(masks, labels):
                class_name = self.class_names[label]
                color = np.array(self.colors.get(class_name, (255, 255, 255))) / 255.0

                # Overlay de máscara
                mask_bool = mask.astype(bool)
                overlay = np.zeros_like(image, dtype=np.float32)
                overlay[mask_bool] = color
                ax.imshow(overlay, alpha=0.4)

        # Dibujar boxes
        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1

            class_name = self.class_names[label]
            color = np.array(self.colors.get(class_name, (255, 255, 255))) / 255.0

            # Rectángulo
            rect = patches.Rectangle(
                (x1, y1), w, h, linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)

            # Etiqueta
            text_parts = [class_name]
            if scores is not None:
                text_parts.append(f"{scores[i]:.2f}")
            if areas is not None:
                text_parts.append(f"{areas[i]:.1f}m²")

            text = " | ".join(text_parts)

            ax.text(
                x1,
                y1 - 5,
                text,
                bbox=dict(facecolor=color, alpha=0.8),
                fontsize=10,
                color="white",
            )

        ax.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Visualización guardada en: {save_path}")

        plt.show()

    def plot_area_distribution(
        self, areas: np.ndarray, labels: np.ndarray, save_path: Optional[str] = None
    ):
        """
        Gráfico de distribución de áreas por tipo de habitación

        Args:
            areas: Áreas en m² (N,)
            labels: Etiquetas de clase (N,)
            save_path: Ruta para guardar
        """
        # Agrupar por tipo
        areas_by_type = {}
        for area, label in zip(areas, labels):
            room_type = self.class_names[label]
            if room_type not in areas_by_type:
                areas_by_type[room_type] = []
            areas_by_type[room_type].append(area)

        # Crear gráfico
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Gráfico de barras
        room_types = list(areas_by_type.keys())
        avg_areas = [np.mean(areas_by_type[rt]) for rt in room_types]
        colors_plot = [
            np.array(self.colors.get(rt, (255, 255, 255))) / 255.0 for rt in room_types
        ]

        ax1.barh(room_types, avg_areas, color=colors_plot)
        ax1.set_xlabel("Área promedio (m²)")
        ax1.set_title("Área Promedio por Tipo de Habitación")
        ax1.grid(True, alpha=0.3)

        # Box plot
        data_for_box = [areas_by_type[rt] for rt in room_types]
        bp = ax2.boxplot(data_for_box, labels=room_types, patch_artist=True, vert=True)

        for patch, color in zip(bp["boxes"], colors_plot):
            patch.set_facecolor(color)

        ax2.set_ylabel("Área (m²)")
        ax2.set_title("Distribución de Áreas por Tipo")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    def create_summary_report(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
        masks: np.ndarray,
        areas: np.ndarray,
        scores: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """
        Reporte visual completo con métricas

        Args:
            image: Imagen original
            boxes, labels, masks, areas, scores: Resultados de detección
            save_path: Ruta para guardar
        """
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Imagen con detecciones
        ax1 = fig.add_subplot(gs[0, :2])
        img_with_detections = self.draw_combined(
            image, boxes, labels, masks, scores, areas
        )
        ax1.imshow(img_with_detections)
        ax1.set_title("Detecciones y Segmentación", fontsize=14, fontweight="bold")
        ax1.axis("off")

        # 2. Estadísticas
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis("off")

        stats_text = "📊 ESTADÍSTICAS\n" + "=" * 30 + "\n\n"
        stats_text += f"Total de habitaciones: {len(labels)}\n"
        stats_text += f"Área total: {np.sum(areas):.1f} m²\n"
        stats_text += f"Área promedio: {np.mean(areas):.1f} m²\n"
        stats_text += f"Confianza promedio: {np.mean(scores):.2%}\n\n"

        stats_text += "Por tipo de habitación:\n" + "-" * 30 + "\n"

        # Contar por tipo
        for class_idx, class_name in enumerate(self.class_names):
            mask_class = labels == class_idx
            count = np.sum(mask_class)

            if count > 0:
                avg_area = np.mean(areas[mask_class])
                stats_text += f"{class_name}: {count} ({avg_area:.1f}m²)\n"

        ax2.text(
            0.1,
            0.9,
            stats_text,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # 3. Distribución de áreas
        ax3 = fig.add_subplot(gs[1, 0])
        areas_by_type = {}
        for area, label in zip(areas, labels):
            room_type = self.class_names[label]
            if room_type not in areas_by_type:
                areas_by_type[room_type] = []
            areas_by_type[room_type].append(area)

        room_types = list(areas_by_type.keys())
        avg_areas = [np.mean(areas_by_type[rt]) for rt in room_types]
        colors_plot = [
            np.array(self.colors.get(rt, (255, 255, 255))) / 255.0 for rt in room_types
        ]

        ax3.barh(room_types, avg_areas, color=colors_plot)
        ax3.set_xlabel("Área promedio (m²)")
        ax3.set_title("Áreas por Tipo")
        ax3.grid(True, alpha=0.3)

        # 4. Distribución de confianzas
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(scores, bins=20, color="steelblue", edgecolor="black", alpha=0.7)
        ax4.set_xlabel("Score de Confianza")
        ax4.set_ylabel("Frecuencia")
        ax4.set_title("Distribución de Confianzas")
        ax4.grid(True, alpha=0.3)

        # 5. Leyenda de colores
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis("off")

        legend_elements = []
        for class_name in self.class_names:
            if class_name in areas_by_type:
                color = np.array(self.colors.get(class_name, (255, 255, 255))) / 255.0
                legend_elements.append(
                    patches.Patch(facecolor=color, edgecolor="black", label=class_name)
                )

        ax5.legend(
            handles=legend_elements,
            loc="center",
            fontsize=9,
            title="Leyenda de Colores",
            title_fontsize=10,
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Reporte guardado en: {save_path}")

        plt.show()


if __name__ == "__main__":
    # Test de visualización
    visualizer = FloorPlanVisualizer(
        class_names=["Bedroom", "Kitchen", "Living Room", "Bathroom"]
    )

    # Crear datos de prueba
    test_image = np.ones((512, 512, 3), dtype=np.uint8) * 255
    test_boxes = np.array([[50, 50, 200, 200], [250, 250, 450, 450]])
    test_labels = np.array([0, 1])
    test_scores = np.array([0.95, 0.87])
    test_areas = np.array([22.5, 40.0])

    # Visualizar
    img_result = visualizer.draw_boxes(
        test_image, test_boxes, test_labels, test_scores, test_areas
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(img_result)
    plt.axis("off")
    plt.title("Test de Visualización")
    plt.show()
