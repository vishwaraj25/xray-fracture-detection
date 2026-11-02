"""
Local web app for X-ray fracture analysis.

    python src/app.py            # then open http://127.0.0.1:8000

Images are analysed in memory and never written to disk.
"""
import argparse
from functools import lru_cache

import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from config import BASE_DIR, IMAGES_DIR, METADATA_PATH, RAW_DIR
from inference import FractureAnalyzer

STATIC_DIR = BASE_DIR / "src" / "static"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

app = FastAPI(title="Fracture Analysis", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache(maxsize=1)
def analyzer():
    return FractureAnalyzer()


@lru_cache(maxsize=1)
def samples():
    """A few held-out test images (never used for training) to try the app with."""
    if not METADATA_PATH.exists():
        return {}
    df = pd.read_csv(METADATA_PATH)
    test = df[(df["split"] == "test") & (df["hardware"] == 0)].drop_duplicates("dup_group")
    picks = []
    for region in ["hand", "leg", "hip", "shoulder"]:
        for label in (1, 0):
            pool = test[(test["region"] == region) & (test["fractured"] == label)]
            if len(pool):
                picks.append(pool.sample(1, random_state=7).iloc[0])
    return {
        row["image_id"].removesuffix(".jpg"): {
            "region": row["region"],
            "folder": row["folder"],
            "file": row["image_id"],
            "thumb": row["path"],
            "label": "Fractured" if row["fractured"] else "Not fractured",
        }
        for row in picks
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/info")
def info():
    return analyzer().info


@app.get("/api/samples")
def list_samples():
    return [{"id": key, "region": s["region"], "label": s["label"]} for key, s in samples().items()]


@app.get("/api/samples/{sample_id}/thumb")
def sample_thumb(sample_id: str):
    sample = samples().get(sample_id)
    if not sample:
        raise HTTPException(404, "Unknown sample")
    return FileResponse(IMAGES_DIR / sample["thumb"], media_type="image/png")


@app.get("/api/samples/{sample_id}/file")
def sample_file(sample_id: str):
    sample = samples().get(sample_id)
    if not sample:
        raise HTTPException(404, "Unknown sample")
    path = RAW_DIR / "images" / sample["folder"] / sample["file"]
    return Response(path.read_bytes(), media_type="image/jpeg")


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 50 MB")
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        return await run_in_threadpool(analyzer().analyze, data)
    except ValueError as err:
        raise HTTPException(422, str(err)) from err


def main():
    parser = argparse.ArgumentParser(description="Run the fracture analysis web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    analyzer()  # load the model before the first request
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
