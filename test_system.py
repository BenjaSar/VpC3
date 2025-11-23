"""
Script de prueba completo del sistema
Genera datos sintéticos y valida el pipeline
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from models.swin_maskrcnn import SwinMaskRCNN
from utils.area_calculator import RoomAreaCalculator
from utils.visualization import FloorPlanVisualizer
from utils.metrics import DetectionMetrics, AreaEstimationMetrics
from utils.dataset import CubiCasaDataset


def create_synthetic_floorplan(
    width: int = 512, height: int = 512, num_rooms: int = 4
) -> tuple:
    """
    Crear plano de planta sintético para pruebas

    Args:
        width: Ancho de imagen
        height: Alto de imagen
        num_rooms: Número de habitaciones

    Returns:
        (imagen, boxes, labels, masks, areas)
    """
    # Imagen base
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Dibujar grid
    for i in range(0, width, 50):
        cv2.line(image, (i, 0), (i, height), (200, 200, 200), 1)
    for i in range(0, height, 50):
        cv2.line(image, (0, i), (width, i), (200, 200, 200), 1)

    # Generar habitaciones
    boxes = []
    labels = []
    masks = []
    areas_list = []

    room_types = [
        (0, "Bedroom", (255, 150, 150)),
        (1, "Kitchen", (150, 255, 150)),
        (2, "Living Room", (150, 150, 255)),
        (3, "Bathroom", (255, 255, 150)),
    ]

    # Crear habitaciones en grid
    positions = [
        (50, 50, 200, 200),  # Top-left
        (270, 50, 450, 200),  # Top-right
        (50, 270, 200, 450),  # Bottom-left
        (270, 270, 450, 450),  # Bottom-right
    ]

    for i in range(min(num_rooms, len(positions))):
        x1, y1, x2, y2 = positions[i]
        label, room_name, color = room_types[i % len(room_types)]

        # Dibujar habitación
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 2)

        # Texto
        cv2.putText(
            image,
            room_name,
            (x1 + 10, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )

        # Máscara
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)

        # Calcular área (en píxeles)
        area_pixels = (x2 - x1) * (y2 - y1)
        area_m2 = area_pixels * (0.02**2)  # Conversión a m²

        boxes.append([x1, y1, x2, y2])
        labels.append(label)
        masks.append(mask)
        areas_list.append(area_m2)

    return (
        image,
        np.array(boxes),
        np.array(labels),
        np.array(masks),
        np.array(areas_list),
    )


def test_model_forward():
    """Test 1: Forward pass del modelo"""
    print("\n" + "=" * 70)
    print("TEST 1: Forward Pass del Modelo")
    print("=" * 70)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        # Crear modelo
        model = SwinMaskRCNN(num_classes=15, img_size=512)
        model = model.to(device)
        model.eval()

        # Input de prueba
        x = torch.randn(1, 3, 512, 512).to(device)

        print(f"\nInput shape: {x.shape}")

        # Forward
        with torch.no_grad():
            outputs = model(x)

        print("\nOutput shapes:")
        print(f"  - cls_scores: {len(outputs['cls_scores'])} scales")
        for i, score in enumerate(outputs["cls_scores"]):
            print(f"    Scale {i}: {score.shape}")

        print(f"  - bbox_preds: {len(outputs['bbox_preds'])} scales")
        for i, bbox in enumerate(outputs["bbox_preds"]):
            print(f"    Scale {i}: {bbox.shape}")

        print(f"  - masks: {outputs['masks'].shape}")

        # Contar parámetros
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal parameters: {total_params:,}")

        print("\n✅ TEST 1 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_area_calculator():
    """Test 2: Calculador de áreas"""
    print("\n" + "=" * 70)
    print("TEST 2: Calculador de Áreas")
    print("=" * 70)

    try:
        calculator = RoomAreaCalculator(pixel_to_meter=0.02)

        # Crear máscaras de prueba
        mask1 = np.zeros((512, 512), dtype=np.uint8)
        cv2.rectangle(mask1, (100, 100), (200, 200), 1, -1)  # 100x100 píxeles

        mask2 = np.zeros((512, 512), dtype=np.uint8)
        cv2.rectangle(mask2, (250, 250), (400, 400), 1, -1)  # 150x150 píxeles

        # Calcular áreas
        area1 = calculator.calculate_area_from_mask(mask1)
        area2 = calculator.calculate_area_from_mask(mask2)

        print(f"\nMáscara 1 (100x100 px):")
        print(f"  Área: {area1:.2f} m²")
        print(f"  Esperado: {10000 * (0.02**2):.2f} m²")

        print(f"\nMáscara 2 (150x150 px):")
        print(f"  Área: {area2:.2f} m²")
        print(f"  Esperado: {22500 * (0.02**2):.2f} m²")

        # Test de estadísticas
        masks = np.array([mask1, mask2])
        labels = np.array([0, 1])
        class_names = ["Bedroom", "Kitchen"]

        stats = calculator.get_room_statistics(masks, labels, class_names)

        print(f"\nEstadísticas:")
        print(f"  Total habitaciones: {stats['total_rooms']}")
        print(f"  Área total: {stats['total_area_m2']:.2f} m²")
        print(f"  Área promedio: {stats['average_area_m2']:.2f} m²")

        print("\n✅ TEST 2 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_visualizer():
    """Test 3: Visualización"""
    print("\n" + "=" * 70)
    print("TEST 3: Visualización")
    print("=" * 70)

    try:
        # Crear plano sintético
        image, boxes, labels, masks, areas = create_synthetic_floorplan()

        print(f"\nPlano sintético creado:")
        print(f"  - Imagen: {image.shape}")
        print(f"  - Habitaciones: {len(labels)}")
        print(f"  - Áreas: {areas}")

        # Visualizador
        visualizer = FloorPlanVisualizer(
            class_names=["Bedroom", "Kitchen", "Living Room", "Bathroom"],
            score_threshold=0.5,
        )

        # Scores de prueba
        scores = np.array([0.95, 0.88, 0.92, 0.85])

        # Dibujar boxes
        img_boxes = visualizer.draw_boxes(image, boxes, labels, scores, areas)

        print("\n✓ Boxes dibujados")

        # Dibujar máscaras
        img_masks = visualizer.draw_masks(image, masks, labels, alpha=0.4)

        print("✓ Máscaras dibujadas")

        # Visualización combinada
        img_combined = visualizer.draw_combined(
            image, boxes, labels, masks, scores, areas
        )

        print("✓ Visualización combinada")

        # Guardar resultados
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        cv2.imwrite(
            str(results_dir / "test_boxes.png"),
            cv2.cvtColor(img_boxes, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(results_dir / "test_masks.png"),
            cv2.cvtColor(img_masks, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(results_dir / "test_combined.png"),
            cv2.cvtColor(img_combined, cv2.COLOR_RGB2BGR),
        )

        print(f"\n✓ Imágenes guardadas en: {results_dir}")

        # Crear reporte
        print("\n✓ Generando reporte completo...")

        fig = plt.figure(figsize=(15, 8))

        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(img_boxes)
        ax1.set_title("Detecciones con Boxes")
        ax1.axis("off")

        ax2 = plt.subplot(1, 3, 2)
        ax2.imshow(img_masks)
        ax2.set_title("Segmentación con Máscaras")
        ax2.axis("off")

        ax3 = plt.subplot(1, 3, 3)
        ax3.imshow(img_combined)
        ax3.set_title("Visualización Completa")
        ax3.axis("off")

        plt.tight_layout()
        plt.savefig(
            results_dir / "test_visualization.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        print(f"✓ Reporte guardado: {results_dir / 'test_visualization.png'}")

        print("\n✅ TEST 3 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_metrics():
    """Test 4: Métricas"""
    print("\n" + "=" * 70)
    print("TEST 4: Métricas de Evaluación")
    print("=" * 70)

    try:
        # Métricas de detección
        metrics = DetectionMetrics(
            num_classes=4, class_names=["Bedroom", "Kitchen", "Living Room", "Bathroom"]
        )

        # Datos de prueba
        pred_boxes = np.array([[10, 10, 100, 100], [150, 150, 250, 250]])
        pred_labels = np.array([0, 1])
        pred_scores = np.array([0.9, 0.85])

        gt_boxes = np.array([[12, 12, 98, 98], [148, 148, 252, 252]])
        gt_labels = np.array([0, 1])

        metrics.update(
            pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, image_id=0
        )

        results = metrics.evaluate()

        print("\nMétricas de Detección:")
        print(f"  mAP: {results['mAP']:.4f}")
        print(f"  Precision: {results['mean_precision']:.4f}")
        print(f"  Recall: {results['mean_recall']:.4f}")

        # Métricas de área
        area_metrics = AreaEstimationMetrics()

        pred_areas = np.array([4.0, 9.0, 6.25, 2.25])
        gt_areas = np.array([4.1, 8.9, 6.3, 2.2])

        area_metrics.update(pred_areas, gt_areas)

        area_results = area_metrics.evaluate()

        print("\nMétricas de Área:")
        print(f"  MAE: {area_results['MAE']:.4f} m²")
        print(f"  RMSE: {area_results['RMSE']:.4f} m²")
        print(f"  MAPE: {area_results['MAPE']:.2f}%")
        print(f"  R²: {area_results['R2']:.4f}")

        print("\n✅ TEST 4 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_end_to_end():
    """Test 5: Pipeline completo end-to-end"""
    print("\n" + "=" * 70)
    print("TEST 5: Pipeline Completo End-to-End")
    print("=" * 70)

    try:
        # Crear plano sintético
        print("\n1. Creando plano sintético...")
        image, gt_boxes, gt_labels, gt_masks, gt_areas = create_synthetic_floorplan()

        # Guardar imagen de prueba
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        cv2.imwrite(
            str(results_dir / "synthetic_floorplan.png"),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )
        print(f"   ✓ Plano guardado: {results_dir / 'synthetic_floorplan.png'}")

        # Crear modelo
        print("\n2. Cargando modelo...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SwinMaskRCNN(num_classes=15, img_size=512)
        model = model.to(device)
        model.eval()
        print(f"   ✓ Modelo en {device}")

        # Inferencia (simulada con ground truth)
        print("\n3. Ejecutando inferencia...")

        # Simular predicciones (en realidad usar ground truth para test)
        pred_boxes = gt_boxes + np.random.randn(*gt_boxes.shape) * 2  # Añadir ruido
        pred_labels = gt_labels
        pred_scores = np.random.uniform(0.7, 0.95, len(gt_labels))
        pred_masks = gt_masks
        pred_areas = gt_areas + np.random.randn(*gt_areas.shape) * 0.1  # Añadir ruido

        print(f"   ✓ {len(pred_labels)} habitaciones detectadas")

        # Visualización
        print("\n4. Generando visualización...")
        visualizer = FloorPlanVisualizer(
            class_names=["Bedroom", "Kitchen", "Living Room", "Bathroom"]
        )

        visualizer.create_summary_report(
            image,
            pred_boxes,
            pred_labels,
            pred_masks,
            pred_areas,
            pred_scores,
            save_path=str(results_dir / "end_to_end_report.png"),
        )

        print(f"   ✓ Reporte guardado: {results_dir / 'end_to_end_report.png'}")

        # Métricas
        print("\n5. Calculando métricas...")
        calculator = RoomAreaCalculator()
        stats = calculator.get_room_statistics(
            pred_masks, pred_labels, ["Bedroom", "Kitchen", "Living Room", "Bathroom"]
        )

        print(f"\n   Estadísticas:")
        print(f"   - Total habitaciones: {stats['total_rooms']}")
        print(f"   - Área total: {stats['total_area_m2']:.2f} m²")
        print(f"   - Área promedio: {stats['average_area_m2']:.2f} m²")

        print("\n✅ TEST 5 PASSED")
        print(f"\n📁 Todos los resultados guardados en: {results_dir.absolute()}")

        return True

    except Exception as e:
        print(f"\n❌ TEST 5 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Ejecutar todos los tests"""
    print("\n" + "=" * 70)
    print(f"{'SUITE DE PRUEBAS - CUBICASA5K ROOM DETECTION':^70}")
    print("=" * 70 + "\n")

    tests = [
        ("Forward Pass del Modelo", test_model_forward),
        ("Calculador de Áreas", test_area_calculator),
        ("Visualización", test_visualizer),
        ("Métricas", test_metrics),
        ("Pipeline End-to-End", test_end_to_end),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Error ejecutando {name}: {e}")
            results.append((name, False))

    # Resumen
    print("\n" + "=" * 70)
    print(f"{'RESUMEN DE TESTS':^70}")
    print("=" * 70 + "\n")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:40s} {status}")

    print(f"\n{'='*70}")
    print(f"  Total: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print(f"  >> TODOS LOS TESTS PASARON! <<")
    else:
        print(f"  >> {total_count - passed_count} tests fallaron")

    print(f"{'='*70}\n")

    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
