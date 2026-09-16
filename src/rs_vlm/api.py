from __future__ import annotations

import io
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from rs_vlm.inference import InferenceEngine


MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/zjt/qwen3vl")
ADAPTER_PATH = os.getenv("ADAPTER_PATH") or None
engine = InferenceEngine(MODEL_PATH, ADAPTER_PATH)
app = FastAPI(title="Remote Sensing Qwen3-VL", version="0.1.0")


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "adapter_path": ADAPTER_PATH,
        "model_loaded": engine.loaded,
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    question: str = Form(..., min_length=1, max_length=2000),
) -> dict[str, object]:
    try:
        payload = await image.read()
        loaded_image = Image.open(io.BytesIO(payload)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="invalid image") from exc
    try:
        return engine.generate_payload(loaded_image, question)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    import uvicorn

    uvicorn.run("rs_vlm.api:app", host=os.getenv("API_HOST", "0.0.0.0"), port=int(os.getenv("API_PORT", "8000")))


if __name__ == "__main__":
    main()

