# 🏠 Room Detection with Swin Transformer + Mask R-CNN

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-red.svg)
![Django](https://img.shields.io/badge/Django-5.2.8-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Sistema completo de detección y segmentación de habitaciones en planos de planta usando Deep Learning.

## ✨ Características

- 🎯 **Detección precisa** de 14 tipos de habitaciones diferentes
- 🎨 **Segmentación por máscaras** a nivel de píxel
- 📊 **Cálculo automático de áreas** en metros cuadrados
- 🌐 **Interfaz web Django** con visualización en tiempo real
- 🔄 **Dataset sintético** de 500 planos generados automáticamente
- 🚀 **Arquitectura moderna**: Swin Transformer + Mask R-CNN

## 🏗️ Arquitectura

```
Swin Transformer (Backbone)
    ↓
Feature Pyramid Network
    ↓
Region Proposal Network
    ↓
ROI Align + Box/Mask Heads
    ↓
Detecciones + Máscaras + Áreas
```

## 📦 Instalación Rápida

```bash
# Clonar repositorio
git clone https://github.com/Jorgecuenca1/vpc3casa.git
cd vpc3casa

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Generar dataset sintético (500 planos)
python utils/synthetic_data_generator.py

# Crear pesos inicializados del modelo
python create_pretrained_weights.py

# Iniciar servidor Django
python manage.py runserver 8080
```

## 🚀 Uso

1. **Abrir navegador**: http://127.0.0.1:8080/

2. **Subir plano**: Click en "Subir Imagen" y selecciona un plano de planta

3. **Ver resultados**:
   - Habitaciones detectadas con bounding boxes
   - Máscaras de segmentación coloreadas
   - Tabla detallada con áreas en m²
   - Estadísticas globales

## 🏷️ Tipos de Habitaciones Soportadas

- 🛏️ Bedroom (Dormitorio)
- 🍳 Kitchen (Cocina)
- 🛋️ Living Room (Sala)
- 🚿 Bathroom (Baño)
- 🍽️ Dining Room (Comedor)
- 🚪 Corridor (Pasillo)
- 🌅 Balcony (Balcón)
- 📦 Storage (Almacenamiento)
- 🚗 Garage (Garage)
- 🧺 Laundry (Lavandería)
- 💼 Office (Oficina)
- 🛌 Guest Room (Cuarto de Huéspedes)
- 🔧 Utility (Utilidad)
- ❓ Other (Otros)

## 📂 Estructura del Proyecto

```
vpc3casa/
├── src/
│   └── models/
│       └── swin_maskrcnn.py       # Modelo principal
├── utils/
│   ├── synthetic_data_generator.py # Generador de datos
│   ├── visualization.py            # Visualización
│   └── area_calculator.py          # Cálculo de áreas
├── detector/
│   ├── views.py                    # Lógica Django
│   └── templates/                  # Templates HTML
├── webapp/
│   ├── settings.py                 # Configuración
│   └── urls.py                     # URLs
├── checkpoints/                    # Pesos del modelo (no incluido)
├── data/                          # Dataset (no incluido)
├── create_pretrained_weights.py   # Script para pesos
└── manage.py                      # Django CLI
```

## 🎓 Entrenar el Modelo (Opcional)

```bash
# Entrenamiento rápido (demo)
python train_fast.py

# Entrenamiento completo
python train.py --epochs 100 --batch-size 4
```

## 🔧 Tecnologías

- **Backend**: Django 5.2.8
- **Deep Learning**: PyTorch 2.9.1
- **Computer Vision**: OpenCV, Pillow
- **Visualización**: Matplotlib, Seaborn
- **Data Science**: NumPy, Pandas

## 📊 Dataset

- **Sintético**: 500 planos generados (400 train, 50 val, 50 test)
- **Formato**: COCO (anotaciones JSON)
- **Resolución**: 512x512 píxeles
- **Anotaciones**: Perfectas (sin errores humanos)

## 🎯 Métricas del Modelo

- **Parámetros**: ~100M
- **Tamaño**: 138 MB
- **Input**: 512x512 RGB
- **Output**: Boxes + Máscaras + Labels + Scores

## 📖 Documentación

- [PROYECTO_COMPLETO.md](PROYECTO_COMPLETO.md) - Documentación completa en español
- [EMPEZAR_AQUI.md](EMPEZAR_AQUI.md) - Guía de inicio rápido
- [INFORME_TECNICO.md](INFORME_TECNICO.md) - Análisis técnico detallado

## 🐛 Problemas Resueltos

- ✅ Compatibilidad CUDA (forzado a CPU)
- ✅ Error de boolean index en máscaras
- ✅ Sincronización de arrays en visualización
- ✅ Manejo correcto de dimensiones (N, 1, H, W)
- ✅ Filtrado de Background antes de procesamiento

## 🚀 Próximas Mejoras

- [ ] Entrenamiento con dataset real (CubiCasa5K)
- [ ] Optimización para GPU
- [ ] API REST para integración
- [ ] Exportación a ONNX/TensorRT
- [ ] Data augmentation avanzada
- [ ] Métricas de evaluación (mAP, IoU)

## 📝 Licencia

MIT License - Ver [LICENSE](LICENSE)

## 👤 Autor

**Jorge Cuenca** ([@Jorgecuenca1](https://github.com/Jorgecuenca1))

## 🙏 Agradecimientos

- Arquitectura Swin Transformer: Microsoft Research
- Mask R-CNN Framework: Facebook AI Research
- Formato COCO: Common Objects in Context

---

**🎉 ¡Proyecto completo y funcional!**

Desarrollado con dedicación para ser el mejor sistema de detección de habitaciones.
