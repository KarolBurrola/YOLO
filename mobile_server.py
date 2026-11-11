import base64
import os
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from ultralytics import YOLO
from util import read_license_plate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="ALPR Sonora - Móvil")

print("Cargando detector de vehículos y placas...")
coco_model = YOLO(os.path.join(BASE_DIR, 'yolov8n.pt') if os.path.exists(os.path.join(BASE_DIR, 'yolov8n.pt')) else 'yolov8n.pt')
license_plate_detector = YOLO(os.path.join(BASE_DIR, 'license_plate_detector.pt'))
print("Modelos listos.")

HTML_UI = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="theme-color" content="#000000">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="ALPR Sonora">
    <link rel="manifest" href="/manifest.json">
    <title>ALPR Sonora - out.mp4 Style</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: #000000;
            color: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            overflow-x: hidden;
        }
        header {
            width: 100%;
            background: #111111;
            padding: 10px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #222222;
        }
        .title { font-size: 1rem; font-weight: 800; color: #00ff00; letter-spacing: 1px; }
        .badge {
            font-size: 0.75rem;
            background: #222222;
            color: #00ff00;
            padding: 4px 8px;
            border-radius: 9999px;
            font-weight: 800;
        }
        .camera-container {
            position: relative;
            width: 100%;
            max-width: 600px;
            max-height: 62vh;
            background: #000;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
            aspect-ratio: 4 / 3;
        }
        video {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        canvas#overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        .plate-banner {
            width: calc(100% - 24px);
            max-width: 540px;
            background: #ffffff;
            border: 3px solid #00ff00;
            border-radius: 8px;
            margin: 12px;
            padding: 12px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0, 255, 0, 0.25);
        }
        .plate-label { font-size: 0.75rem; text-transform: uppercase; color: #444444; font-weight: 700; letter-spacing: 1px; }
        .plate-number {
            font-size: 2.2rem;
            font-weight: 900;
            letter-spacing: 4px;
            color: #000000;
            margin: 2px 0;
            font-family: Arial, "Helvetica Neue", sans-serif;
        }
        .plate-conf { font-size: 0.8rem; color: #059669; font-weight: 700; }
        .history-card {
            width: 100%;
            max-width: 540px;
            padding: 0 16px 20px 16px;
        }
        .history-title { font-size: 0.8rem; color: #888888; margin-bottom: 6px; font-weight: 700; text-transform: uppercase; }
        .history-list {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .history-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #18181b;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 1rem;
            font-weight: 800;
            font-family: monospace;
            border-left: 4px solid #00ff00;
            color: #ffffff;
        }
        .history-time { font-size: 0.75rem; color: #71717a; font-weight: 500; }
    </style>
</head>
<body>
    <header>
        <span class="title">ALPR VISION</span>
        <span class="badge" id="statusBadge">CONECTANDO</span>
    </header>

    <div class="camera-container" id="cameraContainer">
        <video id="webcam" autoplay playsinline muted></video>
        <canvas id="overlay"></canvas>
    </div>

    <div class="plate-banner">
        <div class="plate-label">MATRÍCULA ACTIVA</div>
        <div class="plate-number" id="livePlate">------</div>
        <div class="plate-conf" id="liveConf">Apunta hacia el vehículo</div>
    </div>

    <div class="history-card">
        <div class="history-title">Registro Reciente</div>
        <div class="history-list" id="historyList"></div>
    </div>

    <canvas id="hiddenCanvas" style="display:none;"></canvas>

    <script>
        const cameraContainer = document.getElementById('cameraContainer');
        const video = document.getElementById('webcam');
        const overlay = document.getElementById('overlay');
        const hiddenCanvas = document.getElementById('hiddenCanvas');
        const livePlate = document.getElementById('livePlate');
        const liveConf = document.getElementById('liveConf');
        const statusBadge = document.getElementById('statusBadge');
        const historyList = document.getElementById('historyList');

        const seenPlates = new Set();
        let ws;
        let inFlight = false;
        let lastSentAt = 0;
        let everConnected = false;

        // Preloaded images cache for live render
        const imageCache = {};

        function connectWS() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws`);

            ws.onopen = () => {
                everConnected = true;
                inFlight = false;
                statusBadge.innerText = 'EN VIVO';
                statusBadge.style.background = '#00ff00';
                statusBadge.style.color = '#000000';
                liveConf.innerText = 'Apunta hacia el vehículo';
            };

            ws.onclose = () => {
                inFlight = false;
                statusBadge.innerText = 'OFFLINE';
                statusBadge.style.background = '#ff0000';
                statusBadge.style.color = '#ffffff';
                if (!everConnected && location.protocol === 'https:') {
                    liveConf.innerHTML = 'Si Safari bloquea la conexión en iPhone: <a href="/cert.cer" style="color:#0284c7;text-decoration:underline;">descarga e instala el certificado SSL</a>';
                }
                setTimeout(connectWS, 2000);
            };

            ws.onmessage = (event) => {
                inFlight = false;
                const data = JSON.parse(event.data);
                renderOutMp4Style(data);

                if (data.plate) {
                    livePlate.innerText = data.plate;
                    liveConf.innerText = `CONFIANZA: ${(data.score * 100).toFixed(0)}%`;
                    addHistory(data.plate);
                }
            };
        }

        function addHistory(plate) {
            if (seenPlates.has(plate)) return;
            seenPlates.add(plate);
            const now = new Date().toLocaleTimeString();
            const div = document.createElement('div');
            div.className = 'history-item';
            const plateSpan = document.createElement('span');
            plateSpan.textContent = plate;
            const timeSpan = document.createElement('span');
            timeSpan.className = 'history-time';
            timeSpan.textContent = now;
            div.appendChild(plateSpan);
            div.appendChild(timeSpan);
            historyList.prepend(div);
            if (historyList.children.length > 5) {
                historyList.removeChild(historyList.lastChild);
            }
        }

        // Draw HUD Corner Brackets around Car (Identical to draw_border in visualize.py)
        function drawCarBorder(ctx, x1, y1, x2, y2) {
            ctx.strokeStyle = '#00ff00'; // Pure bright green
            ctx.lineWidth = 6;
            const w = x2 - x1;
            const h = y2 - y1;
            const lineX = Math.min(60, w * 0.3);
            const lineY = Math.min(60, h * 0.3);

            ctx.beginPath();
            // Top-Left
            ctx.moveTo(x1, y1 + lineY); ctx.lineTo(x1, y1); ctx.lineTo(x1 + lineX, y1);
            // Bottom-Left
            ctx.moveTo(x1, y2 - lineY); ctx.lineTo(x1, y2); ctx.lineTo(x1 + lineX, y2);
            // Top-Right
            ctx.moveTo(x2 - lineX, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + lineY);
            // Bottom-Right
            ctx.moveTo(x2 - lineX, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - lineY);
            ctx.stroke();
        }

        // Render EXACT out.mp4 visual overlay
        function renderOutMp4Style(data) {
            const ctx = overlay.getContext('2d');
            overlay.width = video.videoWidth || 640;
            overlay.height = video.videoHeight || 480;
            ctx.clearRect(0, 0, overlay.width, overlay.height);

            const srcW = data.frame_width || 480.0;
            const srcH = data.frame_height || 360.0;
            const scaleX = overlay.width / srcW;
            const scaleY = overlay.height / srcH;

            const detections = data.detections || [];
            detections.forEach(item => {
                const px1 = item.plate_bbox[0] * scaleX;
                const py1 = item.plate_bbox[1] * scaleY;
                const px2 = item.plate_bbox[2] * scaleX;
                const py2 = item.plate_bbox[3] * scaleY;

                // 1. Draw Green Corner Borders for Car (if car detected)
                let carX1 = px1 - 40, carY1 = py1 - 60, carX2 = px2 + 40, carY2 = py2 + 40;
                if (item.car_bbox) {
                    carX1 = item.car_bbox[0] * scaleX;
                    carY1 = item.car_bbox[1] * scaleY;
                    carX2 = item.car_bbox[2] * scaleX;
                    carY2 = item.car_bbox[3] * scaleY;
                }
                drawCarBorder(ctx, carX1, carY1, carX2, carY2);

                // 2. Draw Red Solid Rectangle on License Plate (Identical to visualize.py: cv2.rectangle(..., (0, 0, 255), 12))
                ctx.strokeStyle = '#ff0000'; // Solid bright red
                ctx.lineWidth = 4;
                ctx.strokeRect(px1, py1, px2 - px1, py2 - py1);

                // 3. Floating Placard above car: White box + Bold Black text + Enlarged cropped plate
                if (item.text) {
                    const text = item.text;
                    const centerX = (carX1 + carX2) / 2;
                    const badgeWidth = Math.max(160, (px2 - px1) * 1.8);
                    const badgeHeight = 42;
                    const cropHeight = 36;
                    
                    // Position floating above car top edge
                    let badgeY = carY1 - badgeHeight - cropHeight - 15;
                    if (badgeY < 10) badgeY = Math.max(10, py1 - badgeHeight - cropHeight - 15);
                    const startX = centerX - badgeWidth / 2;

                    // 3.1 White background rectangle (like in out.mp4: (255, 255, 255))
                    ctx.fillStyle = '#ffffff';
                    ctx.fillRect(startX, badgeY, badgeWidth, badgeHeight);

                    // 3.2 Crisp bold black text (like in out.mp4: font_scale, (0, 0, 0))
                    ctx.fillStyle = '#000000';
                    ctx.font = '900 24px Arial, "Helvetica Neue", sans-serif';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(text, centerX, badgeY + badgeHeight / 2);

                    // 3.3 Cropped plate image placed directly beneath white box
                    if (item.crop_b64) {
                        if (!imageCache[text]) {
                            const img = new Image();
                            img.src = 'data:image/jpeg;base64,' + item.crop_b64;
                            imageCache[text] = img;
                        }
                        const cImg = imageCache[text];
                        if (cImg.complete) {
                            ctx.drawImage(cImg, startX, badgeY + badgeHeight, badgeWidth, cropHeight);
                            ctx.strokeStyle = '#222222';
                            ctx.lineWidth = 2;
                            ctx.strokeRect(startX, badgeY + badgeHeight, badgeWidth, cropHeight);
                        }
                    }
                }
            });
        }

        // Start Camera
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            statusBadge.innerText = 'REQUIERE HTTPS';
            liveConf.innerText = 'Abre esta página usando HTTPS para habilitar la cámara en iPhone/Android.';
        } else {
            navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { ideal: 'environment' },
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: false
            }).then(async stream => {
                video.srcObject = stream;
                video.onloadedmetadata = () => {
                    if (video.videoWidth > 0 && video.videoHeight > 0) {
                        cameraContainer.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
                    }
                };
                await video.play().catch(() => {});
                connectWS();
            }).catch(err => {
                statusBadge.innerText = 'SIN CAMARA';
                liveConf.innerText = 'Error al abrir cámara: ' + err.message;
            });
        }

        // Frame Sender Loop (adaptive, avoids queuing frames on mobile)
        setInterval(() => {
            if (!ws || ws.readyState !== WebSocket.OPEN) return;
            if (video.videoWidth === 0 || video.videoHeight === 0) return;
            const now = Date.now();
            if (inFlight && now - lastSentAt < 2500) return;

            const maxDim = 640;
            const scale = Math.min(1, maxDim / Math.max(video.videoWidth, video.videoHeight));
            const sendW = Math.max(160, Math.round(video.videoWidth * scale));
            const sendH = Math.max(120, Math.round(video.videoHeight * scale));

            hiddenCanvas.width = sendW;
            hiddenCanvas.height = sendH;
            const hCtx = hiddenCanvas.getContext('2d');
            hCtx.drawImage(video, 0, 0, sendW, sendH);
            const dataUrl = hiddenCanvas.toDataURL('image/jpeg', 0.65);
            inFlight = true;
            lastSentAt = now;
            ws.send(dataUrl.split(',')[1]);
        }, 200);
    </script>
</body>
</html>
"""

@app.get("/manifest.json")
def get_manifest():
    return {
        "name": "ALPR Sonora Vision",
        "short_name": "ALPR Sonora",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#000000",
        "theme_color": "#000000",
        "orientation": "portrait"
    }


@app.get("/cert.cer")
def download_cert():
    cert_path = os.path.join(BASE_DIR, "cert.pem")
    if os.path.exists(cert_path):
        return FileResponse(
            cert_path,
            media_type="application/x-x509-ca-cert",
            filename="ALPR_Sonora_Cert.cer"
        )
    return HTMLResponse("<h3>No hay certificado SSL local configurado.</h3>", status_code=404)


@app.get("/")
def index():
    return HTMLResponse(HTML_UI)

@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            img_bytes = base64.b64decode(data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # 1. Detect vehicles (car, motorcycle, bus, truck)
            vehicles_bboxes = []
            car_results = coco_model(frame, verbose=False)[0]
            for d in car_results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = d
                if int(class_id) in [2, 3, 5, 7]:
                    vehicles_bboxes.append([x1, y1, x2, y2])

            # 2. Detect license plates
            results = license_plate_detector(frame, verbose=False)[0]
            detections_payload = []
            detected_plate = None
            detected_score = 0.0

            for box in results.boxes.data.tolist():
                px1, py1, px2, py2, score, _ = box
                plate_crop = frame[int(py1):int(py2), int(px1):int(px2)]

                plate_text, p_score = read_license_plate(plate_crop)

                # Match plate to surrounding vehicle bounding box
                matched_car = None
                for cb in vehicles_bboxes:
                    cx1, cy1, cx2, cy2 = cb
                    if px1 >= cx1 and py1 >= cy1 and px2 <= cx2 and py2 <= cy2:
                        matched_car = [int(cx1), int(cy1), int(cx2), int(cy2)]
                        break
                if matched_car is None and len(vehicles_bboxes) > 0:
                    matched_car = [int(v) for v in vehicles_bboxes[0]]

                # Encode crop to base64 jpeg to render identical to out.mp4
                crop_b64 = None
                if plate_crop is not None and getattr(plate_crop, 'size', 0) > 0:
                    ret, buffer = cv2.imencode('.jpg', plate_crop)
                    if ret:
                        crop_b64 = base64.b64encode(buffer).decode('utf-8')

                detections_payload.append({
                    "plate_bbox": [int(px1), int(py1), int(px2), int(py2)],
                    "car_bbox": matched_car,
                    "text": plate_text or "",
                    "crop_b64": crop_b64
                })

                if plate_text:
                    detected_plate = plate_text
                    detected_score = float(p_score)

            await websocket.send_json({
                "detections": detections_payload,
                "plate": detected_plate,
                "score": detected_score,
                "frame_width": int(frame.shape[1]),
                "frame_height": int(frame.shape[0]),
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Error en websocket: {e}")


if __name__ == "__main__":
    import uvicorn

    cert_file = os.path.join(BASE_DIR, "cert.pem")
    key_file = os.path.join(BASE_DIR, "key.pem")
    ssl_kwargs = {}
    if os.path.exists(cert_file) and os.path.exists(key_file):
        ssl_kwargs = {"ssl_certfile": cert_file, "ssl_keyfile": key_file}
        print("Iniciando servidor en HTTPS (puerto 8000)...")
    else:
        print("Iniciando servidor en HTTP (puerto 8000)...")

    uvicorn.run(app, host="0.0.0.0", port=8000, **ssl_kwargs)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
