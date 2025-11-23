from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
import os
import sys
import numpy as np
from PIL import Image
import cv2
import torch

# Agregar path del proyecto para imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Imports del modelo y utilidades
from src.models.swin_maskrcnn import SwinMaskRCNN
from utils.area_calculator import RoomAreaCalculator
from utils.visualization import FloorPlanVisualizer

# Nombres de clases CubiCasa5K
CLASS_NAMES = [
    'Background', 'Bedroom', 'Kitchen', 'Living Room', 'Bathroom',
    'Dining Room', 'Corridor', 'Balcony', 'Storage', 'Garage',
    'Laundry', 'Office', 'Guest Room', 'Utility', 'Other'
]

# Modelo global (cargado una vez)
_model = None
_device = None

def get_model():
    """
    Carga el modelo una vez y lo mantiene en memoria
    """
    global _model, _device

    if _model is None:
        # Forzar CPU para evitar errores de CUDA con Quadro P1000
        _device = torch.device('cpu')
        print(f"Usando device: {_device} (CPU forzado para compatibilidad)")
        print("[INFO] GPU deshabilitada temporalmente - usando CPU para evitar errores CUDA")

        # Crear modelo
        _model = SwinMaskRCNN(num_classes=len(CLASS_NAMES))

        # Intentar cargar pesos entrenados si existen
        checkpoint_path = os.path.join(BASE_DIR, 'checkpoints', 'best_model.pth')
        if os.path.exists(checkpoint_path):
            print(f"Cargando modelo entrenado desde: {checkpoint_path}")
            _model.load_state_dict(torch.load(checkpoint_path, map_location=_device))
        else:
            print("[AVISO] Modelo sin entrenar - usando inicializacion aleatoria para demo")

        _model.to(_device)
        _model.eval()

    return _model, _device


def index(request):
    """
    Vista principal para subir imagenes
    """
    return render(request, 'detector/index.html')


def detect(request):
    """
    Vista para procesar imagen con modelo REAL de Swin Transformer + Mask R-CNN
    """
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Guardar imagen subida
            uploaded_image = request.FILES['image']
            image_path = default_storage.save(f'uploads/{uploaded_image.name}', uploaded_image)
            image_full_path = default_storage.path(image_path)

            # Cargar imagen
            img = cv2.imread(image_full_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_height, original_width = img.shape[:2]

            # Obtener modelo
            model, device = get_model()

            # IMPORTANTE: Resize a 512x512 (tamano esperado por el modelo)
            # El modelo tiene positional embeddings fijos para 512x512
            target_size = 512
            img_resized = cv2.resize(img_rgb, (target_size, target_size))

            # Preparar imagen para el modelo
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # Inferencia
            with torch.no_grad():
                predictions = model(img_tensor)

            # Procesar predicciones
            pred = predictions[0]

            # Extraer detecciones (filtrar por confianza > 0.5)
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            boxes = pred['boxes'].cpu().numpy()
            masks = pred['masks'].cpu().numpy() if 'masks' in pred else None

            # DEBUG: Mostrar predicciones del modelo
            print(f"[DEBUG] Total predicciones generadas: {len(scores)}")
            print(f"[DEBUG] Labels: {labels[:10] if len(labels) > 0 else 'vacio'}")
            print(f"[DEBUG] Scores: {scores[:10] if len(scores) > 0 else 'vacio'}")
            print(f"[DEBUG] Scores max: {scores.max() if len(scores) > 0 else 0}")
            print(f"[DEBUG] Scores min: {scores.min() if len(scores) > 0 else 0}")

            # Filtrar por confianza y limitar número de detecciones
            # NOTA: Threshold bajo para demo con modelo sin entrenar
            threshold = 0.01  # 1% - muy bajo para capturar predicciones del modelo sin entrenar
            keep_idx = scores > threshold

            print(f"[DEBUG] Predicciones con score > {threshold}: {np.sum(keep_idx)}")

            # Aplicar filtro inicial
            boxes = boxes[keep_idx]
            labels = labels[keep_idx]
            scores = scores[keep_idx]
            if masks is not None:
                masks = masks[keep_idx]

            # Limitar a top 100 detecciones para evitar sobrecarga
            MAX_DETECTIONS = 100
            if len(scores) > MAX_DETECTIONS:
                print(f"[DEBUG] Limitando de {len(scores)} a {MAX_DETECTIONS} detecciones (top scores)")
                # Obtener indices de top scores
                top_indices = np.argsort(scores)[-MAX_DETECTIONS:]
                boxes = boxes[top_indices]
                labels = labels[top_indices]
                scores = scores[top_indices]
                if masks is not None:
                    masks = masks[top_indices]

            print(f"[DEBUG] Despues filtro - Total detecciones: {len(labels)}")
            print(f"[DEBUG] Labels: {labels[:10] if len(labels) > 0 else 'vacio'}")

            # Escalar boxes de 512x512 a dimensiones originales
            scale_x = original_width / target_size
            scale_y = original_height / target_size
            boxes[:, [0, 2]] *= scale_x  # x coordinates
            boxes[:, [1, 3]] *= scale_y  # y coordinates

            # Escalar máscaras a dimensiones originales
            if masks is not None and len(masks) > 0:
                print(f"[DEBUG] Redimensionando {len(masks)} mascaras de {masks[0].shape} a ({original_height}, {original_width})")
                masks_resized = []
                for idx, mask in enumerate(masks):
                    try:
                        # Asegurar que mask[0] sea numpy array float32
                        mask_data = mask[0].astype(np.float32)

                        # Verificar dimensiones válidas
                        if mask_data.shape[0] > 0 and mask_data.shape[1] > 0:
                            mask_resized = cv2.resize(
                                mask_data,
                                (original_width, original_height),
                                interpolation=cv2.INTER_LINEAR
                            )
                            masks_resized.append(mask_resized)
                        else:
                            print(f"[WARNING] Mascara {idx} tiene dimensiones invalidas: {mask_data.shape}")
                            # Crear máscara vacía
                            masks_resized.append(np.zeros((original_height, original_width), dtype=np.float32))
                    except Exception as e:
                        print(f"[ERROR] Error redimensionando mascara {idx}: {str(e)}")
                        # Crear máscara vacía en caso de error
                        masks_resized.append(np.zeros((original_height, original_width), dtype=np.float32))

                if len(masks_resized) > 0:
                    masks = np.array(masks_resized)[:, np.newaxis, :, :]
                else:
                    masks = None
                    print("[WARNING] No se pudieron redimensionar mascaras")

            # Filtrar Background (label 0) ANTES de procesar
            # Esto asegura que todos los arrays tengan el mismo tamaño
            non_bg_mask = labels != 0
            boxes_filtered = boxes[non_bg_mask]
            labels_filtered = labels[non_bg_mask]
            scores_filtered = scores[non_bg_mask]
            masks_filtered = masks[non_bg_mask] if masks is not None else None

            print(f"[DEBUG] Despues filtrar background: {len(boxes_filtered)} detecciones")

            # Calcular áreas
            area_calculator = RoomAreaCalculator()
            detections = []
            areas_list = []  # Lista paralela a boxes_filtered
            total_area = 0
            rooms_by_type = {}

            for i, (box, label, score) in enumerate(zip(boxes_filtered, labels_filtered, scores_filtered)):
                room_type = CLASS_NAMES[label] if label < len(CLASS_NAMES) else 'Unknown'

                # Calcular área
                if masks_filtered is not None and i < len(masks_filtered):
                    # Extraer máscara correctamente
                    try:
                        # masks tiene forma (N, 1, H, W)
                        mask_data = masks_filtered[i]  # Obtener máscara i: (1, H, W)

                        # Si tiene dimensión extra, eliminarla
                        if mask_data.ndim == 3 and mask_data.shape[0] == 1:
                            mask_data = mask_data[0]  # Ahora (H, W)
                        elif mask_data.ndim == 3:
                            # Si por alguna razón tiene múltiples canales, tomar el primero
                            mask_data = mask_data[0]

                        # Binarizar
                        mask_binary = mask_data > 0.5
                        area_m2 = area_calculator.calculate_area_from_mask(mask_binary)
                    except Exception as e:
                        print(f"[ERROR] Error procesando mascara {i}: {str(e)}, shape: {masks_filtered[i].shape if i < len(masks_filtered) else 'N/A'}")
                        # Fallback: usar bbox
                        box_width = max(1, int(box[2] - box[0]))
                        box_height = max(1, int(box[3] - box[1]))
                        area_m2 = area_calculator.calculate_area_from_mask(
                            np.ones((box_height, box_width), dtype=bool)
                        )
                else:
                    # Área aproximada desde bbox
                    box_width = max(1, int(box[2] - box[0]))
                    box_height = max(1, int(box[3] - box[1]))
                    area_m2 = area_calculator.calculate_area_from_mask(
                        np.ones((box_height, box_width), dtype=bool)
                    )

                detections.append({
                    'room_type': room_type,
                    'confidence': f'{score*100:.1f}%',
                    'area': f'{area_m2:.1f} m²',
                    'bbox': f'({int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])})'
                })

                areas_list.append(area_m2)  # Guardar área para visualizador
                total_area += area_m2
                rooms_by_type[room_type] = rooms_by_type.get(room_type, 0) + 1

            # Convertir areas_list a numpy array
            areas_array = np.array(areas_list)

            # Si no hay detecciones, generar mensaje
            if len(detections) == 0:
                detections.append({
                    'room_type': 'No detections',
                    'confidence': 'N/A',
                    'area': '0.0 m²',
                    'bbox': 'N/A'
                })
                num_rooms = 0
                total_area = 0
                avg_area = 0
            else:
                num_rooms = len(detections)
                avg_area = total_area / num_rooms

            # Visualizar resultados
            visualizer = FloorPlanVisualizer(
                class_names=CLASS_NAMES,
                score_threshold=0.01  # Bajo threshold para modelo sin entrenar
            )

            # Crear imagen con detecciones (usar arrays FILTRADOS)
            result_image = visualizer.draw_combined(
                img_rgb,
                boxes_filtered,  # Arrays sin Background
                labels_filtered,
                masks_filtered if masks_filtered is not None else None,
                scores_filtered,
                areas_array  # Array de áreas del mismo tamaño
            )

            # Guardar imagen de resultado
            result_filename = f"result_{uploaded_image.name}"
            result_path = os.path.join(settings.MEDIA_ROOT, 'results', result_filename)
            os.makedirs(os.path.dirname(result_path), exist_ok=True)

            result_image_pil = Image.fromarray(result_image)
            result_image_pil.save(result_path)

            result_image_url = os.path.join(settings.MEDIA_URL, 'results', result_filename)

            # Preparar estadísticas
            stats = {
                'total_rooms': num_rooms,
                'total_area_m2': total_area,
                'average_area_m2': avg_area,
                'rooms_by_type': rooms_by_type
            }

            # URLs para las imágenes
            uploaded_image_url = default_storage.url(image_path)

            context = {
                'uploaded_image': uploaded_image_url,
                'result_image': result_image_url,
                'detections': detections,
                'stats': stats,
                'model_type': 'Swin Transformer + Mask R-CNN',
                'device': str(device)
            }

            return render(request, 'detector/results.html', context)

        except Exception as e:
            import traceback
            error_msg = f"Error en deteccion: {str(e)}"
            # No imprimir traceback para evitar problemas de encoding
            return render(request, 'detector/index.html', {'error': error_msg})

    return render(request, 'detector/index.html', {'error': 'No se recibió ninguna imagen'})
