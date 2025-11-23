"""
Script de descarga y preparación del dataset CubiCasa5K
Requiere API de Kaggle configurada
"""

import os
import sys
import zipfile
import shutil
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def setup_kaggle_api():
    """
    Configurar API de Kaggle

    Instrucciones:
    1. Crear cuenta en Kaggle.com
    2. Ir a Account > Create New API Token
    3. Descargar kaggle.json
    4. Colocar en C:/Users/USERNAME/.kaggle/kaggle.json (Windows)
       o ~/.kaggle/kaggle.json (Linux/Mac)
    5. chmod 600 ~/.kaggle/kaggle.json (Linux/Mac)
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        print(f"❌ Error configurando API de Kaggle: {e}")
        print("\nPor favor sigue estos pasos:")
        print("1. Crea una cuenta en https://www.kaggle.com")
        print("2. Ve a tu perfil > Account > Create New API Token")
        print("3. Descarga el archivo kaggle.json")
        print("4. Colócalo en:")
        print("   - Windows: C:/Users/USERNAME/.kaggle/kaggle.json")
        print("   - Linux/Mac: ~/.kaggle/kaggle.json")
        print("5. (Linux/Mac) Ejecuta: chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)


def download_cubicasa5k(data_root="./data"):
    """
    Descargar dataset CubiCasa5K desde Kaggle

    Args:
        data_root: Directorio raíz para datos
    """
    print("\n" + "=" * 60)
    print("📥 DESCARGA DE CUBICASA5K DATASET")
    print("=" * 60)

    # Crear directorio
    data_path = Path(data_root)
    data_path.mkdir(parents=True, exist_ok=True)

    # Configurar API
    print("\n1. Configurando API de Kaggle...")
    api = setup_kaggle_api()
    print("   ✓ API configurada correctamente")

    # Descargar dataset
    print("\n2. Descargando CubiCasa5K...")
    print("   (Esto puede tardar varios minutos dependiendo de tu conexión)")

    dataset_name = "qmarva/cubicasa5k"
    download_path = str(data_path)

    try:
        api.dataset_download_files(
            dataset_name, path=download_path, unzip=True, quiet=False
        )
        print("   ✓ Descarga completada")
    except Exception as e:
        print(f"   ❌ Error descargando dataset: {e}")
        sys.exit(1)

    # Verificar archivos
    print("\n3. Verificando archivos descargados...")
    cubicasa_path = data_path / "cubicasa5k"

    expected_files = ["images", "annotations"]

    all_found = True
    for item in expected_files:
        item_path = cubicasa_path / item
        if item_path.exists():
            print(f"   ✓ {item}")
        else:
            print(f"   ❌ {item} (no encontrado)")
            all_found = False

    if not all_found:
        print("\n⚠️  Algunos archivos no se encontraron.")
        print("   El dataset podría tener una estructura diferente.")
        print("   Verifica manualmente en:", cubicasa_path)

    return cubicasa_path


def create_dataset_splits(cubicasa_path, train_ratio=0.7, val_ratio=0.15):
    """
    Crear splits de train/val/test

    Args:
        cubicasa_path: Path al dataset
        train_ratio: Proporción de training
        val_ratio: Proporción de validación
    """
    print("\n4. Creando splits de datos...")

    annotations_path = cubicasa_path / "annotations"
    images_path = cubicasa_path / "images"

    if not annotations_path.exists() or not images_path.exists():
        print("   ⚠️  No se encontraron carpetas de anotaciones o imágenes")
        print("   Salteando creación de splits...")
        return

    # Listar todas las imágenes
    image_files = list(images_path.glob("*.png")) + list(images_path.glob("*.jpg"))
    total_images = len(image_files)

    print(f"   Total de imágenes: {total_images}")

    if total_images == 0:
        print("   ⚠️  No se encontraron imágenes")
        return

    # Mezclar aleatoriamente
    import random

    random.seed(42)
    random.shuffle(image_files)

    # Calcular splits
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)

    train_files = image_files[:train_size]
    val_files = image_files[train_size : train_size + val_size]
    test_files = image_files[train_size + val_size :]

    print(f"   - Train: {len(train_files)} imágenes")
    print(f"   - Val: {len(val_files)} imágenes")
    print(f"   - Test: {len(test_files)} imágenes")

    # Crear directorios de splits
    for split in ["train", "val", "test"]:
        (cubicasa_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (cubicasa_path / "annotations" / split).mkdir(parents=True, exist_ok=True)

    # Mover archivos
    print("\n   Organizando archivos...")

    for i, (split_name, files) in enumerate(
        [("train", train_files), ("val", val_files), ("test", test_files)]
    ):
        for file in tqdm(files, desc=f"   {split_name}", ncols=70):
            # Copiar imagen
            dest_img = cubicasa_path / "images" / split_name / file.name
            if not dest_img.exists():
                shutil.copy(file, dest_img)

    print("   ✓ Splits creados correctamente")


def create_coco_annotations(cubicasa_path):
    """
    Crear anotaciones en formato COCO si no existen

    Args:
        cubicasa_path: Path al dataset
    """
    print("\n5. Verificando formato de anotaciones...")

    annotations_path = cubicasa_path / "annotations"

    # Verificar si ya existen anotaciones COCO
    coco_files = list(annotations_path.glob("*.json"))

    if len(coco_files) > 0:
        print(f"   ✓ Se encontraron {len(coco_files)} archivos de anotaciones")
        return

    print("   ⚠️  No se encontraron anotaciones en formato COCO")
    print("   Necesitarás convertir las anotaciones manualmente")
    print("   o descargar un dataset pre-procesado.")


def verify_installation(cubicasa_path):
    """
    Verificar que el dataset esté correctamente instalado

    Args:
        cubicasa_path: Path al dataset
    """
    print("\n" + "=" * 60)
    print("✅ VERIFICACIÓN DE INSTALACIÓN")
    print("=" * 60)

    checks = {
        "Dataset directory": cubicasa_path.exists(),
        "Images folder": (cubicasa_path / "images").exists(),
        "Annotations folder": (cubicasa_path / "annotations").exists(),
    }

    all_passed = True
    for check, passed in checks.items():
        status = "✓" if passed else "❌"
        print(f"   {status} {check}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 Dataset instalado correctamente!")
        print(f"\nRuta del dataset: {cubicasa_path.absolute()}")
    else:
        print("\n⚠️  Algunos checks fallaron. Revisa la instalación.")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Descargar y preparar dataset CubiCasa5K"
    )
    parser.add_argument(
        "--data-root", type=str, default="./data", help="Directorio raíz para datos"
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Saltar descarga (solo procesar)"
    )

    args = parser.parse_args()

    print("\n" + "🏠" * 20)
    print("CubiCasa5K Dataset Downloader")
    print("🏠" * 20)

    # Descargar dataset
    if not args.skip_download:
        cubicasa_path = download_cubicasa5k(args.data_root)
    else:
        cubicasa_path = Path(args.data_root) / "cubicasa5k"
        print(f"\n⏭️  Salteando descarga. Usando: {cubicasa_path}")

    # Crear splits
    create_dataset_splits(cubicasa_path)

    # Verificar anotaciones
    create_coco_annotations(cubicasa_path)

    # Verificar instalación
    verify_installation(cubicasa_path)

    print("\n" + "=" * 60)
    print("📚 SIGUIENTES PASOS")
    print("=" * 60)
    print("\n1. Instalar dependencias:")
    print("   pip install -r requirements.txt")
    print("\n2. Entrenar modelo:")
    print("   python train.py --config configs/swin_maskrcnn_cubicasa.py")
    print("\n3. Ejecutar inferencia:")
    print("   python inference.py --checkpoint best_model.pth --image test.png")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
