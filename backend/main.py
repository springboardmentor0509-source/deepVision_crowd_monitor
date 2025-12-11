import os
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.model_loader import manager
from backend.utility import save_upload_tmpfile, generate_heatmap_overlay

app = FastAPI(title="DeepVision Crowd Monitor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "DeepVision Crowd Monitor API Running"}


@app.get("/models")
def get_models():
    return {"models": list(manager.model_paths.keys())}


@app.post("/predict/{model_name}")
async def predict(model_name: str, file: UploadFile = File(...)):
    if model_name not in manager.model_paths:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    suffix = os.path.splitext(file.filename)[1]
    tmp_path = None

    try:
        # Save uploaded file
        tmp_path = save_upload_tmpfile(file, suffix=suffix)

        # Run prediction
        density, count = manager.predict(model_name, tmp_path)

        # Prepare heatmap overlay
        heatmap_b64 = None
        if density is not None:
            heatmap_bytes = generate_heatmap_overlay(tmp_path, density)
            heatmap_b64 = base64.b64encode(heatmap_bytes).decode()

        # Response (NO heavy density map)
        response = {
            "model": model_name,
            "filename": file.filename,
            "predicted_count": int(count) if count is not None else None,
            "heatmap_image": heatmap_b64,
        }

        return JSONResponse(response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
