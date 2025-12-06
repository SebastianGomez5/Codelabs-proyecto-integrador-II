# Codelabs-proyecto-integrador-II 🚀

Implementación de codelabs desarrollados durante el curso de **Proyecto Integrador 2** en la Universidad del Valle. Este repositorio contiene ejemplos prácticos de visión por computadora, procesamiento de audio, redes neuronales y machine learning.

---

## 📋 Información del Proyecto

**Estudiante**: Juan Sebastian Gomez Agudelo  
**Código**: 2259474  
**Grupo**: 51  
**Universidad**: Universidad del Valle

---

## 📁 Estructura del Proyecto

El proyecto está organizado en 7 codelabs independientes:

### **Codelab 1: MFCC con Micrófono** 🎙️
- **Descripción**: Grabación de audio desde el micrófono y cálculo de coeficientes MFCC (Mel-Frequency Cepstral Coefficients)
- **Ubicación**: `codelab1/mfcc_con_microfono/`
- **Archivo principal**: `record_and_mfcc.py`
- **Tecnologías**: NumPy, Matplotlib, SoundDevice, SoundFile
- **Uso**: Procesamiento de audio para aplicaciones de reconocimiento de voz

### **Codelab 2: Detección de Rostros con MTCNN** 👤
- **Descripción**: Detección de rostros en imágenes usando el modelo Multi-task Cascaded Convolutional Networks (MTCNN)
- **Ubicación**: `codelab2/mtcnn/`
- **Archivo principal**: `carga_imagen.py`
- **Tecnologías**: OpenCV, MTCNN, NumPy, Matplotlib
- **Uso**: Detección de rostros y características faciales en imágenes estáticas

### **Codelab 3: Detección de Rostros en Webcam** 🎥
- **Descripción**: Detección de rostros en tiempo real desde webcam usando MediaPipe
- **Ubicación**: `codelab3/`
- **Archivo principal**: `webcam.py`
- **Tecnologías**: OpenCV, MediaPipe
- **Requisitos**: `mediapipe`, `opencv-python`
- **Uso**: Detección en tiempo real de rostros

### **Codelab 4: Red Neuronal XOR** 🧠
- **Descripción**: Implementación de una red neuronal artificial para resolver el problema clásico XOR usando TensorFlow/Keras
- **Ubicación**: `codelab4/tensorflorXor/`
- **Archivo principal**: `EjemploXor.py`
- **Tecnologías**: TensorFlow, Keras, NumPy
- **Uso**: Demostración de redes neuronales y Deep Learning

### **Codelab 5: Detección YOLO-Lite** ⚡
- **Descripción**: Detección de objetos en tiempo real con YOLOv8 (versión lite) desde webcam e imágenes
- **Ubicación**: `codelab5/yolo-lite/`
- **Archivos principales**:
  - `deteccion-real-webcam.py` - Detección en tiempo real
  - `yolo-lite-image.py` - Detección en imágenes
  - `export-result-json.py` - Exportación de resultados
- **Tecnologías**: Ultralytics YOLO, OpenCV
- **Modelo**: `yolov8n.pt` (YOLO Nano - optimizado)
- **Uso**: Detección rápida de objetos

### **Codelab 6: Comparación SSD vs YOLO** ⚔️
- **Descripción**: Comparación de dos modelos de detección de objetos: SSD (Single Shot MultiBox Detector) y YOLOv8
- **Ubicación**: `codelab6/deteccion-ssd-yolo/`
- **Archivos principales**:
  - `SsdImage.py` - Detección con SSD
  - `resultadosSsd.py` - Análisis de resultados SSD
  - `comparacionYoloLite.py` - Comparación entre modelos
- **Tecnologías**: OpenCV, Ultralytics YOLO
- **Uso**: Análisis comparativo de rendimiento y precisión

### **Codelab 7: Clasificador de Comentarios** 📝
- **Descripción**: Clasificador de sentimientos para comentarios de negocios usando Machine Learning (SVM + TF-IDF)
- **Ubicación**: `codelab7/Clasificador_comentarios/`
- **Archivo principal**: `clasificador-comentarios-negocio.py`
- **Tecnologías**: Scikit-Learn, Pandas, NumPy
- **Modelos guardados**: 
  - `modelo.joblib` - Modelo SVM entrenado
  - `tfidf.joblib` - Vectorizador TF-IDF
- **Uso**: Análisis de sentimientos en opiniones de clientes

---

## 🛠️ Requisitos Previos

- Python 3.7+
- pip (gestor de paquetes de Python)
- Cámara web (para codelabs 3, 5 y 6)
- Micrófono (para codelab 1)

---

## 📦 Instalación

Cada codelab tiene su propio entorno virtual. Para ejecutar cualquier proyecto:

### Opción 1: Usar el entorno virtual existente

```bash
# En el directorio del codelab específico
cd codelab<N>
.\env\Scripts\Activate.ps1  # En Windows PowerShell
# o
source env/Scripts/activate  # En Linux/Mac