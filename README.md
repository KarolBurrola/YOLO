# ALPR Sonora - YOLOv8

Sistema de reconocimiento automático de matrículas (ALPR) adaptado para placas de **Sonora y México** con **YOLOv8**, seguimiento vehicular con **SORT**, estimación de velocidad (**SpeedEstimator**) y servidor web en tiempo real con **FastAPI / WebSockets**.

## Características principales

- **Soporte de placas de Sonora y México (`util.py`)**: Validación y corrección de caracteres confusos de OCR (`0/O`, `1/I`, etc.) para formatos `LLLDDDD` (`WYZ1234`), `LLLDDDL` (`WAB123A`), `LLLDDD` (`WAA123`) y compatibilidad con formato UK (`CE61VYL`).
- **Detección y seguimiento (`main.py`, `sort/`)**: Detección de vehículos (`yolov8n.pt`, descargado automáticamente por Ultralytics) y detección de matrículas (`license_plate_detector.pt`), con tracking mediante SORT.
- **Interpolación y renderizado (`add_missing_data.py`, `visualize.py`)**: Suavizado de cuadros faltantes y generación de video anotado (`out.mp4`).
- **Estimador de velocidad (`speed_estimator.py`)**: Cálculo de velocidad en km/h mediante homografía planar o escala lineal con suavizado exponencial (EMA).
- **Servidor web / móvil (`mobile_server.py`)**: Interfaz web con cámara en vivo vía WebSocket usando FastAPI y Uvicorn.

## Estructura del repositorio

- `main.py`: Pipeline principal de detección, tracking y OCR sobre `./sample.mp4` (genera `test.csv`).
- `util.py`: Motor de lectura EasyOCR y validación/formateo de placas.
- `add_missing_data.py`: Interpola fotogramas faltantes de `test.csv` a `test_interpolated.csv`.
- `visualize.py`: Genera `out.mp4` a partir de `sample.mp4` y `test_interpolated.csv`.
- `speed_estimator.py`: Módulo `SpeedEstimator` para estimación de velocidad vehicular.
- `mobile_server.py`: Servidor FastAPI + WebSocket para transmisión de cámara en tiempo real.
- `sort/`: Implementación de tracking SORT (`abewley/sort`).
- `license_plate_detector.pt`: Pesos entrenados de YOLOv8 para detección de matrículas.
- `test_sonora_plates.py` y `test_speed_estimator.py`: Pruebas unitarias.

## Instalación

```bash
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

## Uso

### 1. Ejecutar pruebas unitarias
```bash
python test_sonora_plates.py
python test_speed_estimator.py
```

### 2. Procesar un video (`sample.mp4`)
Coloca tu video como `sample.mp4` en la raíz del proyecto (`yolov8n.pt` se descargará automáticamente en la primera ejecución):

```bash
python main.py
python add_missing_data.py
python visualize.py
```

### 3. Iniciar el servidor web en tiempo real
```bash
python mobile_server.py
# o con uvicorn directamente:
uvicorn mobile_server:app --host 0.0.0.0 --port 8000
```

> Nota: Para abrir la cámara desde un navegador móvil en red local, los navegadores requieren `localhost` o un contexto `HTTPS`.
