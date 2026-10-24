import io
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import cv2
import imageio_ffmpeg
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from PIL import Image
from ultralytics import YOLO

BACKEND = Path(__file__).resolve().parent
ROOT = BACKEND.parent
MODEL_PATH = BACKEND / "best.pt"
NEW_IMG = ROOT / "storage" / "new_data" / "images"
NEW_LBL = ROOT / "storage" / "new_data" / "labels"
LOG_FILE = ROOT / "storage" / "retrain.log"
for p in (NEW_IMG, NEW_LBL):
    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Gun Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

model = YOLO(str(MODEL_PATH))
train_proc = None
reloaded_after_train = True


def read_image(data: bytes):
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image")


@app.get("/")
def health():
    return {"status": "ok"}


# ---------- IMAGE ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.25):
    img = read_image(await file.read())
    result = model(img, conf=conf, verbose=False)[0]
    dets = [
        {
            "label": model.names[int(b.cls)],
            "confidence": round(float(b.conf), 3),
            "box": [round(x, 1) for x in b.xyxy[0].tolist()],
        }
        for b in result.boxes
    ]
    return {"count": len(dets), "detections": dets}


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...), conf: float = 0.25):
    img = read_image(await file.read())
    result = model(img, conf=conf, verbose=False)[0]
    out = Image.fromarray(result.plot()[..., ::-1])
    buf = io.BytesIO()
    out.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


# ---------- VIDEO ----------
@app.post("/predict/video")
def predict_video(file: UploadFile = File(...), conf: float = 0.25):
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "in.mp4"
    src.write_bytes(file.file.read())

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(400, "Video open nahi hui")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw = tmp / "raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    total, hit = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model(frame, conf=conf, verbose=False)[0]
        total += 1
        if len(res.boxes):
            hit += 1
        writer.write(res.plot())
    cap.release()
    writer.release()

    out = tmp / "out.mp4"
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(raw),
         "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True,
    )
    return FileResponse(
        out, media_type="video/mp4",
        headers={"X-Total-Frames": str(total), "X-Detected-Frames": str(hit)},
        background=BackgroundTask(shutil.rmtree, tmp, True),
    )


# ---------- NAYA DATA SAVE ----------
@app.post("/save-training-image")
async def save_training_image(file: UploadFile = File(...),
                              use_predictions: bool = Form(True)):
    """Image + (optional) model ke predicted boxes ko YOLO label ki tarah save karo"""
    img = read_image(await file.read())
    name = uuid.uuid4().hex
    img.save(NEW_IMG / f"{name}.jpg", "JPEG")

    lines = []
    if use_predictions:
        result = model(img, conf=0.5, verbose=False)[0]   # strict threshold
        for b, xywhn in zip(result.boxes, result.boxes.xywhn.tolist()):
            x, y, w, h = xywhn
            lines.append(f"{int(b.cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    (NEW_LBL / f"{name}.txt").write_text("\n".join(lines))
    return {"saved": name, "boxes": len(lines)}


# ---------- RETRAIN ----------
@app.post("/retrain")
def retrain(epochs: int = 30):
    global train_proc, reloaded_after_train
    if train_proc is not None and train_proc.poll() is None:
        raise HTTPException(400, "Training pehle se chal rahi hai")
    if not list(NEW_IMG.glob("*.jpg")):
        raise HTTPException(400, "Naye images nahi hain")

    log = open(LOG_FILE, "w", encoding="utf-8")
    train_proc = subprocess.Popen(
        [sys.executable, str(BACKEND / "retrain.py"), str(epochs)],
        stdout=log, stderr=subprocess.STDOUT,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    reloaded_after_train = False
    return {"status": "training started"}


@app.get("/retrain/status")
def retrain_status():
    global model, reloaded_after_train
    running = train_proc is not None and train_proc.poll() is None
    if train_proc is not None and not running and not reloaded_after_train:
        model = YOLO(str(MODEL_PATH))      # agar best.pt replace hua to naya load
        reloaded_after_train = True
    log = LOG_FILE.read_text(encoding="utf-8", errors="ignore")[-1500:] if LOG_FILE.exists() else ""
    return {
        "running": running,
        "new_images": len(list(NEW_IMG.glob("*.jpg"))),
        "log": log,
    }


@app.post("/reload-model")
def reload_model():
    global model
    model = YOLO(str(MODEL_PATH))
    return {"status": "reloaded"}