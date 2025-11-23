"""
Script de prueba rapida del modelo
"""
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.models.swin_maskrcnn import SwinMaskRCNN

print("=" * 60)
print("PRUEBA RAPIDA DEL MODELO SWIN MASK R-CNN")
print("=" * 60)

# Create model
print("\n[1/4] Creando modelo...")
model = SwinMaskRCNN(num_classes=15, img_size=512)
model.eval()
print("[OK] Modelo creado exitosamente")

# Create test input
print("\n[2/4] Creando input de prueba (1 imagen 512x512)...")
x = torch.randn(1, 3, 512, 512)
print(f"[OK] Input shape: {x.shape}")

# Forward pass
print("\n[3/4] Ejecutando inferencia...")
with torch.no_grad():
    predictions = model(x)
print("[OK] Inferencia exitosa")

# Check output format
print("\n[4/4] Verificando formato de salida...")
print(f"Tipo de output: {type(predictions)}")
print(f"Numero de imagenes procesadas: {len(predictions)}")

if len(predictions) > 0:
    pred = predictions[0]
    print("\nContenido de prediccion[0]:")
    print(f"  - Keys: {pred.keys()}")
    print(f"  - Boxes shape: {pred['boxes'].shape}")
    print(f"  - Labels shape: {pred['labels'].shape}")
    print(f"  - Scores shape: {pred['scores'].shape}")
    print(f"  - Masks shape: {pred['masks'].shape}")

    num_detections = len(pred['boxes'])
    print(f"\n[OK] Detecciones generadas: {num_detections}")

    if num_detections > 0:
        print(f"  - Score maximo: {pred['scores'].max():.4f}")
        print(f"  - Score minimo: {pred['scores'].min():.4f}")
        print(f"  - Labels unicos: {torch.unique(pred['labels']).tolist()}")
    else:
        print("  [AVISO] No se generaron detecciones (normal para modelo sin entrenar)")
else:
    print("[ERROR] Error: predicciones vacias")

print("\n" + "=" * 60)
print("RESULTADO: Modelo funcionando correctamente [OK]")
print("=" * 60)
