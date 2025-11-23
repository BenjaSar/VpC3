"""
Generador de planos de planta sintéticos para entrenamiento
Crea imágenes de planos con habitaciones y anotaciones perfectas
"""

import numpy as np
import cv2
from pathlib import Path
import json
import random
from typing import List, Dict, Tuple
from tqdm import tqdm


class SyntheticFloorPlanGenerator:
    """
    Generador de planos sintéticos con anotaciones
    """

    ROOM_TYPES = [
        'Bedroom', 'Kitchen', 'Living Room', 'Bathroom',
        'Dining Room', 'Corridor', 'Balcony', 'Storage',
        'Garage', 'Laundry', 'Office', 'Guest Room', 'Utility', 'Other'
    ]

    # Tamaños típicos de habitaciones en píxeles (a escala)
    ROOM_SIZES = {
        'Bedroom': (80, 120, 100, 150),  # min_w, max_w, min_h, max_h
        'Kitchen': (60, 100, 60, 100),
        'Living Room': (120, 180, 100, 160),
        'Bathroom': (40, 70, 40, 70),
        'Dining Room': (80, 120, 80, 120),
        'Corridor': (30, 50, 80, 150),
        'Balcony': (50, 80, 30, 60),
        'Storage': (30, 60, 30, 60),
        'Garage': (100, 150, 80, 120),
        'Laundry': (40, 70, 40, 70),
        'Office': (70, 110, 70, 110),
        'Guest Room': (80, 120, 90, 130),
        'Utility': (40, 70, 40, 70),
        'Other': (50, 100, 50, 100),
    }

    def __init__(self, img_size: int = 512, wall_thickness: int = 3):
        self.img_size = img_size
        self.wall_thickness = wall_thickness

    def generate_floor_plan(self) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generar un plano de planta sintético

        Returns:
            image: Imagen del plano (512x512, RGB)
            annotations: Lista de anotaciones [{'bbox', 'label', 'mask'}]
        """
        # Crear imagen base (fondo blanco, paredes grises)
        image = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 255

        # Máscara para tracking de espacio ocupado
        occupied = np.zeros((self.img_size, self.img_size), dtype=bool)

        annotations = []

        # Número aleatorio de habitaciones (3-8)
        num_rooms = random.randint(3, 8)

        # Generar habitaciones
        for _ in range(num_rooms):
            room_type = random.choice(self.ROOM_TYPES)
            label = self.ROOM_TYPES.index(room_type) + 1  # +1 porque 0 es background

            # Intentar colocar habitación
            placed = False
            attempts = 0
            max_attempts = 50

            while not placed and attempts < max_attempts:
                attempts += 1

                # Generar tamaño aleatorio
                min_w, max_w, min_h, max_h = self.ROOM_SIZES[room_type]
                w = random.randint(min_w, max_w)
                h = random.randint(min_h, max_h)

                # Posición aleatoria
                x = random.randint(10, self.img_size - w - 10)
                y = random.randint(10, self.img_size - h - 10)

                # Verificar si hay espacio
                region = occupied[y:y+h, x:x+w]
                if region.sum() / region.size < 0.3:  # Permitir 30% de overlap
                    # Dibujar habitación
                    self._draw_room(image, x, y, w, h, room_type)

                    # Crear máscara
                    mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
                    cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)

                    # Marcar como ocupado
                    occupied[y:y+h, x:x+w] = True

                    # Guardar anotación
                    annotations.append({
                        'bbox': [x, y, x+w, y+h],
                        'label': label,
                        'mask': mask,
                        'room_type': room_type
                    })

                    placed = True

        return image, annotations

    def _draw_room(self, image: np.ndarray, x: int, y: int, w: int, h: int, room_type: str):
        """
        Dibujar habitación en el plano
        """
        # Color de relleno basado en tipo
        colors = {
            'Bedroom': (230, 230, 250),      # Lavender
            'Kitchen': (255, 250, 205),      # Lemon
            'Living Room': (240, 255, 240),  # Honeydew
            'Bathroom': (240, 248, 255),     # Alice Blue
            'Dining Room': (255, 250, 240),  # Floral White
            'Corridor': (245, 245, 245),     # White Smoke
            'Balcony': (240, 255, 255),      # Azure
            'Storage': (250, 235, 215),      # Antique White
            'Garage': (220, 220, 220),       # Gainsboro
            'Laundry': (240, 248, 255),      # Alice Blue
            'Office': (245, 245, 220),       # Beige
            'Guest Room': (255, 240, 245),   # Lavender Blush
            'Utility': (245, 245, 245),      # White Smoke
            'Other': (255, 255, 255),        # White
        }

        fill_color = colors.get(room_type, (255, 255, 255))

        # Rellenar habitación
        cv2.rectangle(image, (x, y), (x+w, y+h), fill_color, -1)

        # Dibujar paredes
        wall_color = (100, 100, 100)  # Gris oscuro
        cv2.rectangle(image, (x, y), (x+w, y+h), wall_color, self.wall_thickness)

        # Añadir texto del tipo de habitación (pequeño)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        thickness = 1

        # Calcular posición del texto
        text = room_type[:3].upper()  # Abreviar
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = x + w//2 - text_w//2
        text_y = y + h//2 + text_h//2

        cv2.putText(image, text, (text_x, text_y), font, font_scale, (50, 50, 50), thickness)

    def generate_dataset(
        self,
        output_dir: str,
        num_samples: int = 500,
        split: str = 'train'
    ):
        """
        Generar dataset completo

        Args:
            output_dir: Directorio de salida
            num_samples: Número de muestras a generar
            split: 'train', 'val', o 'test'
        """
        output_path = Path(output_dir)
        images_dir = output_path / 'images' / split
        annotations_dir = output_path / 'annotations' / split

        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerando {num_samples} planos sinteticos ({split})...")

        all_annotations = []

        for i in tqdm(range(num_samples), desc=f"Generando {split}"):
            # Generar plano
            image, annotations = self.generate_floor_plan()

            # Guardar imagen
            image_filename = f"floor_plan_{split}_{i:04d}.png"
            image_path = images_dir / image_filename
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # Convertir anotaciones a formato COCO
            image_id = i

            for ann_idx, ann in enumerate(annotations):
                x1, y1, x2, y2 = ann['bbox']

                # COCO format
                coco_ann = {
                    'id': image_id * 100 + ann_idx,
                    'image_id': image_id,
                    'category_id': ann['label'],
                    'bbox': [x1, y1, x2 - x1, y2 - y1],  # COCO: [x, y, width, height]
                    'area': (x2 - x1) * (y2 - y1),
                    'segmentation': self._mask_to_polygon(ann['mask']),
                    'iscrowd': 0
                }
                all_annotations.append(coco_ann)

            # Guardar máscara combinada
            mask_filename = f"floor_plan_{split}_{i:04d}_mask.png"
            mask_path = annotations_dir / mask_filename

            # Crear máscara multi-clase
            combined_mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            for ann in annotations:
                mask_binary = ann['mask'] > 0
                combined_mask[mask_binary] = ann['label']

            cv2.imwrite(str(mask_path), combined_mask)

        # Guardar anotaciones en formato COCO
        coco_format = {
            'info': {
                'description': 'Synthetic Floor Plan Dataset',
                'version': '1.0',
                'year': 2025
            },
            'licenses': [],
            'images': [
                {
                    'id': i,
                    'file_name': f"floor_plan_{split}_{i:04d}.png",
                    'height': self.img_size,
                    'width': self.img_size
                }
                for i in range(num_samples)
            ],
            'annotations': all_annotations,
            'categories': [
                {'id': i+1, 'name': name, 'supercategory': 'room'}
                for i, name in enumerate(self.ROOM_TYPES)
            ]
        }

        annotations_file = output_path / 'annotations' / f'{split}.json'
        with open(annotations_file, 'w') as f:
            json.dump(coco_format, f, indent=2)

        print(f"Dataset generado en: {output_path}")
        print(f"   - Imagenes: {images_dir}")
        print(f"   - Anotaciones: {annotations_file}")

    def _mask_to_polygon(self, mask: np.ndarray) -> List[List[float]]:
        """
        Convertir máscara binaria a polígono COCO
        """
        # Encontrar contornos
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        polygons = []
        for contour in contours:
            if len(contour) >= 3:  # Al menos 3 puntos
                polygon = contour.flatten().tolist()
                polygons.append(polygon)

        return polygons if polygons else [[0, 0, 0, 0, 0, 0]]


def main():
    """
    Generar dataset completo
    """
    print("\n" + "=" * 60)
    print("GENERADOR DE PLANOS SINTETICOS")
    print("=" * 60)

    generator = SyntheticFloorPlanGenerator(img_size=512)
    output_dir = "./data/synthetic"

    # Generar splits
    generator.generate_dataset(output_dir, num_samples=400, split='train')
    generator.generate_dataset(output_dir, num_samples=50, split='val')
    generator.generate_dataset(output_dir, num_samples=50, split='test')

    print("\nDataset sintetico completo generado!")
    print("\nEstadisticas:")
    print("   - Train: 400 planos")
    print("   - Val: 50 planos")
    print("   - Test: 50 planos")
    print("   - Total: 500 planos sinteticos")
    print("\nListo para entrenar!")


if __name__ == "__main__":
    main()
