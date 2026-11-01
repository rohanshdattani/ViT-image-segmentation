import io
import os
import threading

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from transformers import SegformerForSemanticSegmentation

MODEL_ID = "ChillinSpyder/ViT-image-segmentation-model"
LONG_SIDE = 512            # the model was trained on images with a long side of 1024
MAX_UPLOAD_MB = 15
ALLOWED_TYPES = {"image/jpeg", "image/png"}

CLASS_NAMES = ["other", "wall", "road", "vegetation", "vehicle", "roof", "water"]
PALETTE = np.array(
    [(80, 80, 80), (230, 160, 60), (190, 190, 200), (50, 160, 60), (220, 50, 50), (150, 60, 40), (50, 110, 220)],
    dtype=np.uint8,
)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---- model: loaded once at startup, not per request -------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if os.environ.get("TORCH_THREADS"):          # in a container, torch may see the host's cores; pin it to your vCPUs
    torch.set_num_threads(int(os.environ["TORCH_THREADS"]))
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).to(device).eval()
inference_lock = threading.Lock()            # one forward pass at a time (CPU demos get slower if they overlap)
print("Using device:", device)


# ---- inference helpers (plain functions, no web code) -------------------------------------
def prepare_image(content: bytes) -> np.ndarray:
    """Bytes -> normalised float32 array (3, H, W). Keeps the aspect ratio, long side <= 1024 (never upscaled)."""
    try:
        image = Image.open(io.BytesIO(content))
        width, height = image.size
        scale = min(1.0, LONG_SIDE / max(width, height))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image.draft("RGB", size)             # fast, reduced-size JPEG decode (no-op for PNG)
        image = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as err:
        raise ValueError("The file is not a readable image") from err
    array = (np.asarray(image, dtype=np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def segment(content: bytes) -> np.ndarray:
    """Bytes -> class-id mask (H, W), uint8, at the resized input resolution."""
    tensor = torch.from_numpy(prepare_image(content)).unsqueeze(0).to(device)
    with inference_lock, torch.inference_mode():
        logits = model(pixel_values=tensor).logits                     # (1, C, H/4, W/4)
        logits = F.interpolate(logits, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
        mask = logits.argmax(dim=1)[0]
    return mask.cpu().numpy().astype(np.uint8)


# ---- web app ---------------------------------------------------------------------------
app = FastAPI(title="Drone image segmentation")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "classes": CLASS_NAMES}


@app.post("/predict")
async def predict_image(file: UploadFile):
    print(">>> /predict START" flush=True)
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are allowed")
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (limit {MAX_UPLOAD_MB} MB)")
    try:
        print(">>> Starting segmentation" flush=True)
        mask = await run_in_threadpool(segment, content)
        print(">>> Segmentation finished" flush=True)               # keeps the server responsive during inference
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    buffer = io.BytesIO()
    Image.fromarray(PALETTE[mask]).save(buffer, format="PNG")
    print(">>> Sending response" flush=True)
    return Response(content=buffer.getvalue(), media_type="image/png")


# Must stay LAST: it serves static/index.html at "/" and would otherwise shadow routes defined after it.
app.mount("/", StaticFiles(directory="static", html=True), name="static")