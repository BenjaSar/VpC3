"""
CubiCasa5K Dataset Loader
Optimizado para detección de habitaciones y cálculo de áreas
"""

import os
import json
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from pycocotools import mask as maskUtils
from shapely.geometry import Polygon
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CubiCasaDataset(Dataset):
    """
    Dataset personalizado para CubiCasa5K

    Características:
    - Carga de planos de planta con anotaciones de polígonos
    - Generación de máscaras de segmentación
    - Cálculo de áreas de habitaciones
    - Aumento de datos específico para planos arquitectónicos
    """

    # Clases de habitaciones principales en CubiCasa5K
    ROOM_CLASSES = [
        "Bedroom",
        "Kitchen",
        "Living Room",
        "Bathroom",
        "Dining Room",
        "Corridor",
        "Balcony",
        "Storage",
        "Office",
        "Laundry",
        "Garage",
        "Terrace",
        "Closet",
        "Entrance",
        "Other",
    ]

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        img_prefix: str,
        pipeline: Optional[List] = None,
        test_mode: bool = False,
        img_size: Tuple[int, int] = (512, 512),
        pixel_to_meter: float = 0.02,  # Conversión pixel a metros
        filter_empty: bool = True,
    ):
        """
        Args:
            data_root: Directorio raíz del dataset
            ann_file: Archivo de anotaciones (formato COCO JSON)
            img_prefix: Prefijo para rutas de imágenes
            pipeline: Pipeline de preprocesamiento
            test_mode: Modo de evaluación
            img_size: Tamaño de imagen de entrada
            pixel_to_meter: Factor de conversión pixel a metros
            filter_empty: Filtrar imágenes sin anotaciones
        """
        self.data_root = data_root
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.test_mode = test_mode
        self.img_size = img_size
        self.pixel_to_meter = pixel_to_meter
        self.filter_empty = filter_empty

        # Cargar anotaciones
        self.coco = self._load_annotations()
        self.img_ids = self._get_img_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.coco["categories"])}

        # Pipeline de aumento de datos
        if pipeline is None:
            self.transform = self._get_default_transform(test_mode)
        else:
            self.transform = pipeline

    def _load_annotations(self) -> Dict:
        """Cargar anotaciones desde archivo JSON"""
        with open(self.ann_file, "r") as f:
            coco_data = json.load(f)
        return coco_data

    def _get_img_ids(self) -> List[int]:
        """Obtener IDs de imágenes válidas"""
        img_ids = []
        for img_info in self.coco["images"]:
            img_id = img_info["id"]
            if self.filter_empty:
                # Verificar que tenga anotaciones
                ann_ids = self._get_ann_ids(img_id)
                if len(ann_ids) > 0:
                    img_ids.append(img_id)
            else:
                img_ids.append(img_id)
        return img_ids

    def _get_ann_ids(self, img_id: int) -> List[int]:
        """Obtener IDs de anotaciones para una imagen"""
        ann_ids = []
        for ann in self.coco["annotations"]:
            if ann["image_id"] == img_id:
                ann_ids.append(ann["id"])
        return ann_ids

    def _get_default_transform(self, test_mode: bool):
        """Pipeline de aumento de datos para planos de planta"""
        if test_mode:
            return A.Compose(
                [
                    A.Resize(self.img_size[0], self.img_size[1]),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(self.img_size[0], self.img_size[1]),
                    # Aumento específico para planos arquitectónicos
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Rotate(limit=45, p=0.3),
                    A.RandomBrightnessContrast(p=0.3),
                    A.GaussNoise(p=0.2),
                    A.Blur(blur_limit=3, p=0.2),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(
                    format="coco", label_fields=["category_ids"], min_visibility=0.3
                ),
            )

    def _load_image(self, img_id: int) -> np.ndarray:
        """Cargar imagen desde disco"""
        img_info = self._get_img_info(img_id)
        img_path = os.path.join(self.img_prefix, img_info["file_name"])

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"No se pudo cargar imagen: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _get_img_info(self, img_id: int) -> Dict:
        """Obtener información de imagen"""
        for img_info in self.coco["images"]:
            if img_info["id"] == img_id:
                return img_info
        raise ValueError(f"No se encontró imagen con ID: {img_id}")

    def _load_annotations(self, img_id: int) -> Dict:
        """Cargar anotaciones para una imagen"""
        ann_ids = self._get_ann_ids(img_id)

        bboxes = []
        labels = []
        masks = []
        areas = []

        for ann_id in ann_ids:
            ann = self._get_annotation(ann_id)

            # Bbox en formato [x, y, w, h]
            bbox = ann["bbox"]
            bboxes.append(bbox)

            # Etiqueta de clase
            category_id = ann["category_id"]
            label = self.cat2label[category_id]
            labels.append(label)

            # Máscara de segmentación
            if "segmentation" in ann:
                mask = self._polygon_to_mask(
                    ann["segmentation"], self.img_size[0], self.img_size[1]
                )
                masks.append(mask)

                # Calcular área real en m²
                area_pixels = np.sum(mask)
                area_m2 = area_pixels * (self.pixel_to_meter**2)
                areas.append(area_m2)
            else:
                # Área desde bbox si no hay segmentación
                area_pixels = bbox[2] * bbox[3]
                area_m2 = area_pixels * (self.pixel_to_meter**2)
                areas.append(area_m2)
                masks.append(np.zeros(self.img_size, dtype=np.uint8))

        return {
            "bboxes": np.array(bboxes, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int64),
            "masks": np.array(masks, dtype=np.uint8),
            "areas": np.array(areas, dtype=np.float32),
        }

    def _get_annotation(self, ann_id: int) -> Dict:
        """Obtener anotación específica"""
        for ann in self.coco["annotations"]:
            if ann["id"] == ann_id:
                return ann
        raise ValueError(f"No se encontró anotación con ID: {ann_id}")

    def _polygon_to_mask(
        self, segmentation: List, height: int, width: int
    ) -> np.ndarray:
        """
        Convertir polígono a máscara binaria

        Args:
            segmentation: Lista de polígonos [[x1,y1,x2,y2,...]]
            height: Alto de la imagen
            width: Ancho de la imagen

        Returns:
            Máscara binaria (H, W)
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        for polygon in segmentation:
            if isinstance(polygon, list):
                # Formato: [x1, y1, x2, y2, ...]
                polygon = np.array(polygon).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [polygon], 1)

        return mask

    def calculate_room_area(
        self, mask: np.ndarray, pixel_to_meter: Optional[float] = None
    ) -> float:
        """
        Calcular área de habitación desde máscara

        Args:
            mask: Máscara binaria de la habitación
            pixel_to_meter: Factor de conversión (usa self.pixel_to_meter si None)

        Returns:
            Área en metros cuadrados
        """
        if pixel_to_meter is None:
            pixel_to_meter = self.pixel_to_meter

        area_pixels = np.sum(mask > 0)
        area_m2 = area_pixels * (pixel_to_meter**2)
        return area_m2

    def __len__(self) -> int:
        """Número de muestras en el dataset"""
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Dict:
        """
        Obtener muestra del dataset

        Returns:
            Dict con:
                - image: Tensor de imagen (C, H, W)
                - bboxes: Bounding boxes (N, 4)
                - labels: Etiquetas de clase (N,)
                - masks: Máscaras de segmentación (N, H, W)
                - areas: Áreas en m² (N,)
                - img_id: ID de imagen
        """
        img_id = self.img_ids[idx]

        # Cargar imagen
        image = self._load_image(img_id)

        # Cargar anotaciones
        annotations = self._load_annotations(img_id)

        # Aplicar transformaciones
        if not self.test_mode and len(annotations["bboxes"]) > 0:
            transformed = self.transform(
                image=image,
                bboxes=annotations["bboxes"],
                category_ids=annotations["labels"],
            )
            image = transformed["image"]
            bboxes = np.array(transformed["bboxes"], dtype=np.float32)
            labels = np.array(transformed["category_ids"], dtype=np.int64)
        else:
            transformed = self.transform(image=image)
            image = transformed["image"]
            bboxes = annotations["bboxes"]
            labels = annotations["labels"]

        return {
            "image": image,
            "bboxes": (
                torch.from_numpy(bboxes) if len(bboxes) > 0 else torch.zeros((0, 4))
            ),
            "labels": (
                torch.from_numpy(labels)
                if len(labels) > 0
                else torch.zeros((0,), dtype=torch.long)
            ),
            "masks": (
                torch.from_numpy(annotations["masks"])
                if len(annotations["masks"]) > 0
                else torch.zeros((0, *self.img_size))
            ),
            "areas": (
                torch.from_numpy(annotations["areas"])
                if len(annotations["areas"]) > 0
                else torch.zeros((0,))
            ),
            "img_id": img_id,
        }

    def get_cat_ids(self, idx: int) -> List[int]:
        """Obtener IDs de categorías para una imagen"""
        img_id = self.img_ids[idx]
        ann_ids = self._get_ann_ids(img_id)

        cat_ids = []
        for ann_id in ann_ids:
            ann = self._get_annotation(ann_id)
            cat_ids.append(ann["category_id"])

        return cat_ids


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Función de colación personalizada para DataLoader

    Maneja tamaños variables de bboxes/máscaras por imagen
    """
    images = torch.stack([item["image"] for item in batch])

    batch_dict = {
        "images": images,
        "bboxes": [item["bboxes"] for item in batch],
        "labels": [item["labels"] for item in batch],
        "masks": [item["masks"] for item in batch],
        "areas": [item["areas"] for item in batch],
        "img_ids": [item["img_id"] for item in batch],
    }

    return batch_dict


if __name__ == "__main__":
    # Test del dataset
    dataset = CubiCasaDataset(
        data_root="data/cubicasa5k/",
        ann_file="data/cubicasa5k/annotations/train.json",
        img_prefix="data/cubicasa5k/images/train/",
        test_mode=False,
    )

    print(f"Dataset cargado: {len(dataset)} muestras")
    print(f"Clases: {dataset.ROOM_CLASSES}")

    # Cargar primera muestra
    sample = dataset[0]
    print(f"\nMuestra de prueba:")
    print(f"- Imagen: {sample['image'].shape}")
    print(f"- Bboxes: {sample['bboxes'].shape}")
    print(f"- Labels: {sample['labels']}")
    print(f"- Máscaras: {sample['masks'].shape}")
    print(f"- Áreas (m²): {sample['areas']}")
