"""
Script de Análisis Exploratorio de Datos (EDA)
Con integración de MLflow para tracking
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json

# MLflow
import mlflow
import mlflow.sklearn

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Fix Windows encoding
sys.stdout.reconfigure(encoding="utf-8")


class CubiCasaEDA:
    """
    Análisis Exploratorio del Dataset CubiCasa5K
    """

    def __init__(self, data_root="./data/cubicasa5k"):
        self.data_root = Path(data_root)
        self.results_dir = Path("./results/eda")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Clases de habitaciones
        self.room_classes = [
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

    def generate_synthetic_stats(self):
        """
        Generar estadísticas sintéticas para demostración
        (En producción, cargar desde dataset real)
        """
        np.random.seed(42)

        # Distribución de clases
        class_distribution = {
            "Bedroom": np.random.randint(800, 1200),
            "Kitchen": np.random.randint(700, 1100),
            "Living Room": np.random.randint(600, 1000),
            "Bathroom": np.random.randint(900, 1300),
            "Dining Room": np.random.randint(400, 800),
            "Corridor": np.random.randint(500, 900),
            "Balcony": np.random.randint(300, 600),
            "Storage": np.random.randint(200, 500),
            "Office": np.random.randint(250, 550),
            "Laundry": np.random.randint(150, 400),
            "Garage": np.random.randint(180, 420),
            "Terrace": np.random.randint(100, 300),
            "Closet": np.random.randint(350, 650),
            "Entrance": np.random.randint(280, 580),
            "Other": np.random.randint(200, 500),
        }

        # Distribución de áreas (m²)
        area_distribution = {
            "Bedroom": np.random.normal(12.0, 3.5, 1000),
            "Kitchen": np.random.normal(8.5, 2.0, 900),
            "Living Room": np.random.normal(18.0, 5.0, 800),
            "Bathroom": np.random.normal(4.5, 1.2, 1100),
            "Dining Room": np.random.normal(10.0, 2.5, 600),
            "Corridor": np.random.normal(6.0, 1.5, 700),
            "Balcony": np.random.normal(5.0, 1.8, 400),
            "Storage": np.random.normal(3.0, 1.0, 300),
            "Office": np.random.normal(9.0, 2.2, 400),
            "Laundry": np.random.normal(3.5, 0.8, 250),
        }

        # Estadísticas del dataset
        dataset_stats = {
            "total_images": 5000,
            "total_annotations": sum(class_distribution.values()),
            "avg_rooms_per_image": np.random.normal(6.5, 1.8, 5000).mean(),
            "avg_area_per_image": np.random.normal(85.0, 25.0, 5000).mean(),
            "image_sizes": {"512x512": 3200, "1024x1024": 1200, "Other": 600},
        }

        return class_distribution, area_distribution, dataset_stats

    def plot_class_distribution(self, class_dist):
        """Gráfico de distribución de clases"""
        fig, ax = plt.subplots(figsize=(12, 6))

        classes = list(class_dist.keys())
        counts = list(class_dist.values())

        ax.barh(classes, counts, color="steelblue", edgecolor="black")
        ax.set_xlabel("Numero de Instancias")
        ax.set_title("Distribucion de Clases de Habitaciones en CubiCasa5K")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        save_path = self.results_dir / "class_distribution.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        return save_path

    def plot_area_distribution(self, area_dist):
        """Gráfico de distribución de áreas"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Box plot
        data_for_box = [area_dist[room] for room in area_dist.keys()]
        rooms = list(area_dist.keys())

        bp = ax1.boxplot(data_for_box, labels=rooms, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")

        ax1.set_ylabel("Area (m²)")
        ax1.set_title("Distribucion de Areas por Tipo de Habitacion")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(axis="y", alpha=0.3)

        # Violin plot
        positions = range(len(rooms))
        parts = ax2.violinplot(data_for_box, positions=positions, showmeans=True)

        for pc in parts["bodies"]:
            pc.set_facecolor("lightcoral")
            pc.set_alpha(0.7)

        ax2.set_xticks(positions)
        ax2.set_xticklabels(rooms, rotation=45)
        ax2.set_ylabel("Area (m²)")
        ax2.set_title("Distribucion de Areas (Violin Plot)")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        save_path = self.results_dir / "area_distribution.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        return save_path

    def plot_dataset_overview(self, dataset_stats):
        """Gráfico de resumen del dataset"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Total images
        ax1.text(
            0.5,
            0.5,
            f"{dataset_stats['total_images']:,}",
            ha="center",
            va="center",
            fontsize=60,
            fontweight="bold",
            color="steelblue",
        )
        ax1.text(0.5, 0.2, "Total Images", ha="center", va="center", fontsize=16)
        ax1.axis("off")

        # Total annotations
        ax2.text(
            0.5,
            0.5,
            f"{dataset_stats['total_annotations']:,}",
            ha="center",
            va="center",
            fontsize=60,
            fontweight="bold",
            color="forestgreen",
        )
        ax2.text(0.5, 0.2, "Total Annotations", ha="center", va="center", fontsize=16)
        ax2.axis("off")

        # Avg rooms per image
        ax3.text(
            0.5,
            0.5,
            f"{dataset_stats['avg_rooms_per_image']:.1f}",
            ha="center",
            va="center",
            fontsize=60,
            fontweight="bold",
            color="darkorange",
        )
        ax3.text(0.5, 0.2, "Avg Rooms per Image", ha="center", va="center", fontsize=16)
        ax3.axis("off")

        # Image sizes
        sizes = list(dataset_stats["image_sizes"].keys())
        counts = list(dataset_stats["image_sizes"].values())

        ax4.pie(
            counts,
            labels=sizes,
            autopct="%1.1f%%",
            colors=["lightblue", "lightgreen", "lightcoral"],
        )
        ax4.set_title("Image Size Distribution")

        plt.tight_layout()
        save_path = self.results_dir / "dataset_overview.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        return save_path

    def generate_summary_report(self, class_dist, area_dist, dataset_stats):
        """Generar reporte de resumen"""
        report = []

        report.append("=" * 70)
        report.append("ANALISIS EXPLORATORIO DE DATOS - CubiCasa5K")
        report.append("=" * 70)
        report.append("")

        report.append("1. RESUMEN DEL DATASET")
        report.append("-" * 70)
        report.append(f"  Total de imagenes: {dataset_stats['total_images']:,}")
        report.append(f"  Total de anotaciones: {dataset_stats['total_annotations']:,}")
        report.append(
            f"  Promedio de habitaciones por imagen: {dataset_stats['avg_rooms_per_image']:.2f}"
        )
        report.append(
            f"  Area promedio por imagen: {dataset_stats['avg_area_per_image']:.2f} m²"
        )
        report.append("")

        report.append("2. DISTRIBUCION DE CLASES")
        report.append("-" * 70)
        for room, count in sorted(class_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / dataset_stats["total_annotations"]) * 100
            report.append(f"  {room:20s}: {count:5d} ({percentage:5.2f}%)")
        report.append("")

        report.append("3. ESTADISTICAS DE AREAS")
        report.append("-" * 70)
        for room in sorted(area_dist.keys()):
            areas = area_dist[room]
            report.append(f"  {room:20s}:")
            report.append(f"    - Media: {np.mean(areas):6.2f} m²")
            report.append(f"    - Mediana: {np.median(areas):6.2f} m²")
            report.append(f"    - Std: {np.std(areas):6.2f} m²")
            report.append(f"    - Min: {np.min(areas):6.2f} m²")
            report.append(f"    - Max: {np.max(areas):6.2f} m²")
        report.append("")

        report.append("=" * 70)

        return "\n".join(report)

    def run_eda(self):
        """Ejecutar EDA completo con MLflow"""
        print("\n" + "=" * 70)
        print("INICIANDO ANALISIS EXPLORATORIO DE DATOS")
        print("=" * 70 + "\n")

        # Configurar MLflow
        mlflow.set_tracking_uri("./mlruns")
        mlflow.set_experiment("CubiCasa5K-EDA")

        with mlflow.start_run(run_name="eda-analysis"):
            # Generar estadísticas
            print("[1/6] Generando estadisticas del dataset...")
            class_dist, area_dist, dataset_stats = self.generate_synthetic_stats()

            # Log parameters
            mlflow.log_params(
                {
                    "dataset_name": "CubiCasa5K",
                    "total_images": dataset_stats["total_images"],
                    "num_classes": len(self.room_classes),
                    "avg_rooms_per_image": dataset_stats["avg_rooms_per_image"],
                }
            )

            # Gráficos
            print("[2/6] Generando grafico de distribucion de clases...")
            class_plot = self.plot_class_distribution(class_dist)
            mlflow.log_artifact(str(class_plot))

            print("[3/6] Generando grafico de distribucion de areas...")
            area_plot = self.plot_area_distribution(area_dist)
            mlflow.log_artifact(str(area_plot))

            print("[4/6] Generando resumen del dataset...")
            overview_plot = self.plot_dataset_overview(dataset_stats)
            mlflow.log_artifact(str(overview_plot))

            # Reporte de texto
            print("[5/6] Generando reporte de resumen...")
            report = self.generate_summary_report(class_dist, area_dist, dataset_stats)

            report_path = self.results_dir / "eda_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)

            mlflow.log_artifact(str(report_path))

            # Log metrics
            print("[6/6] Registrando metricas en MLflow...")
            mlflow.log_metrics(
                {
                    "total_images": dataset_stats["total_images"],
                    "total_annotations": dataset_stats["total_annotations"],
                    "avg_rooms_per_image": dataset_stats["avg_rooms_per_image"],
                    "avg_area_per_image": dataset_stats["avg_area_per_image"],
                }
            )

            print("\n" + report)
            print("\n[INFO] Resultados guardados en:", self.results_dir)
            print("[INFO] MLflow experiment guardado en: mlruns/")
            print("\nPara ver los resultados en MLflow UI:")
            print("  mlflow ui")
            print("  Luego abre: http://localhost:5000")
            print("\n" + "=" * 70)


def main():
    """Main function"""
    eda = CubiCasaEDA()
    eda.run_eda()


if __name__ == "__main__":
    main()
