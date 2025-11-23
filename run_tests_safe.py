"""
Script de prueba simplificado sin emojis para Windows
"""

import sys
import io

# Fix encoding para Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch
import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")  # Backend sin display
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from utils.area_calculator import RoomAreaCalculator
from utils.visualization import FloorPlanVisualizer
from utils.metrics import DetectionMetrics, AreaEstimationMetrics


def create_synthetic_floorplan(width=512, height=512, num_rooms=4):
    """Crear plano de planta sintético"""
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Grid
    for i in range(0, width, 50):
        cv2.line(image, (i, 0), (i, height), (200, 200, 200), 1)
    for i in range(0, height, 50):
        cv2.line(image, (0, i), (width, i), (200, 200, 200), 1)

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

    positions = [
        (50, 50, 200, 200),
        (270, 50, 450, 200),
        (50, 270, 200, 450),
        (270, 270, 450, 450),
    ]

    for i in range(min(num_rooms, len(positions))):
        x1, y1, x2, y2 = positions[i]
        label, room_name, color = room_types[i % len(room_types)]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(
            image,
            room_name,
            (x1 + 10, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)

        area_pixels = (x2 - x1) * (y2 - y1)
        area_m2 = area_pixels * (0.02**2)

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


def test_area_calculator():
    """Test calculador de áreas"""
    print("\n" + "=" * 70)
    print("TEST: Calculador de Áreas")
    print("=" * 70)

    try:
        calculator = RoomAreaCalculator(pixel_to_meter=0.02)

        mask1 = np.zeros((512, 512), dtype=np.uint8)
        cv2.rectangle(mask1, (100, 100), (200, 200), 1, -1)

        mask2 = np.zeros((512, 512), dtype=np.uint8)
        cv2.rectangle(mask2, (250, 250), (400, 400), 1, -1)

        area1 = calculator.calculate_area_from_mask(mask1)
        area2 = calculator.calculate_area_from_mask(mask2)

        print(f"\nMascara 1 (100x100 px): {area1:.2f} m²")
        print(f"Mascara 2 (150x150 px): {area2:.2f} m²")

        masks = np.array([mask1, mask2])
        labels = np.array([0, 1])
        class_names = ["Bedroom", "Kitchen"]

        stats = calculator.get_room_statistics(masks, labels, class_names)

        print(f"\nEstadisticas:")
        print(f"  Total habitaciones: {stats['total_rooms']}")
        print(f"  Area total: {stats['total_area_m2']:.2f} m²")
        print(f"  Area promedio: {stats['average_area_m2']:.2f} m²")

        print("\n[OK] TEST PASSED\n")
        return True

    except Exception as e:
        print(f"\n[ERROR] TEST FAILED: {e}\n")
        import traceback

        traceback.print_exc()
        return False


def test_visualizer():
    """Test visualización"""
    print("\n" + "=" * 70)
    print("TEST: Visualizacion")
    print("=" * 70)

    try:
        image, boxes, labels, masks, areas = create_synthetic_floorplan()

        print(f"\nPlano sintetico creado:")
        print(f"  - Imagen: {image.shape}")
        print(f"  - Habitaciones: {len(labels)}")

        visualizer = FloorPlanVisualizer(
            class_names=["Bedroom", "Kitchen", "Living Room", "Bathroom"],
            score_threshold=0.5,
        )

        scores = np.array([0.95, 0.88, 0.92, 0.85])

        img_boxes = visualizer.draw_boxes(image, boxes, labels, scores, areas)
        img_masks = visualizer.draw_masks(image, masks, labels, alpha=0.4)
        img_combined = visualizer.draw_combined(
            image, boxes, labels, masks, scores, areas
        )

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        cv2.imwrite(
            str(results_dir / "test_boxes.png"),
            cv2.cvtColor(img_boxes, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(results_dir / "test_combined.png"),
            cv2.cvtColor(img_combined, cv2.COLOR_RGB2BGR),
        )

        print(f"\n[OK] Imagenes guardadas en: {results_dir}")
        print("\n[OK] TEST PASSED\n")
        return True

    except Exception as e:
        print(f"\n[ERROR] TEST FAILED: {e}\n")
        import traceback

        traceback.print_exc()
        return False


def test_metrics():
    """Test métricas"""
    print("\n" + "=" * 70)
    print("TEST: Metricas")
    print("=" * 70)

    try:
        metrics = DetectionMetrics(
            num_classes=4, class_names=["Bedroom", "Kitchen", "Living Room", "Bathroom"]
        )

        pred_boxes = np.array([[10, 10, 100, 100], [150, 150, 250, 250]])
        pred_labels = np.array([0, 1])
        pred_scores = np.array([0.9, 0.85])

        gt_boxes = np.array([[12, 12, 98, 98], [148, 148, 252, 252]])
        gt_labels = np.array([0, 1])

        metrics.update(
            pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, image_id=0
        )

        results = metrics.evaluate()

        print("\nMetricas de Deteccion:")
        print(f"  mAP: {results['mAP']:.4f}")
        print(f"  Precision: {results['mean_precision']:.4f}")
        print(f"  Recall: {results['mean_recall']:.4f}")

        area_metrics = AreaEstimationMetrics()
        pred_areas = np.array([4.0, 9.0, 6.25, 2.25])
        gt_areas = np.array([4.1, 8.9, 6.3, 2.2])

        area_metrics.update(pred_areas, gt_areas)
        area_results = area_metrics.evaluate()

        print("\nMetricas de Area:")
        print(f"  MAE: {area_results['MAE']:.4f} m²")
        print(f"  RMSE: {area_results['RMSE']:.4f} m²")
        print(f"  MAPE: {area_results['MAPE']:.2f}%")

        print("\n[OK] TEST PASSED\n")
        return True

    except Exception as e:
        print(f"\n[ERROR] TEST FAILED: {e}\n")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 70)
    print("      SUITE DE PRUEBAS - CUBICASA5K ROOM DETECTION      ")
    print("=" * 70 + "\n")

    tests = [
        ("Calculador de Areas", test_area_calculator),
        ("Visualizacion", test_visualizer),
        ("Metricas", test_metrics),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n[ERROR] Error ejecutando {name}: {e}\n")
            results.append((name, False))

    print("\n" + "=" * 70)
    print("RESUMEN DE TESTS")
    print("=" * 70 + "\n")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "[OK] PASSED" if passed else "[X] FAILED"
        print(f"  {name:40s} {status}")

    print(f"\n{'='*70}")
    print(f"  Total: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print(f"  >> TODOS LOS TESTS PASARON! <<")
    else:
        print(f"  >> {total_count - passed_count} tests fallaron")

    print(f"{'='*70}\n")

    print("\n[INFO] Resultados guardados en: results/")
    print("[INFO] Revisa las imagenes generadas\n")

    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
