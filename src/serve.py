from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException, Request
from google.cloud import storage
from pydantic import BaseModel
import joblib
import pandas as pd

ARTIFACT_BUCKET = os.environ.get("ARTIFACT_BUCKET")
MODEL_KEY = "artifacts/current/model.joblib"
MODEL_PATH = os.path.expanduser("~/models/model.joblib")
FEATURE_NAMES = [
    "age",
    "workclass",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]


def download_model() -> object:
    """Download and load the currently released model from Cloud Storage."""
    if not ARTIFACT_BUCKET:
        raise RuntimeError("ARTIFACT_BUCKET must be set before starting the API")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    client = storage.Client()
    blob = client.bucket(ARTIFACT_BUCKET).blob(MODEL_KEY)
    blob.download_to_filename(MODEL_PATH)
    print(f"Model downloaded from gs://{ARTIFACT_BUCKET}/{MODEL_KEY}")
    return joblib.load(MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup. A missing model prevents the service from reporting
    # healthy, which lets the CI deployment health check catch failed releases.
    app.state.model = download_model()
    yield


app = FastAPI(lifespan=lifespan)


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest, request: Request):
    if len(req.features) != len(FEATURE_NAMES):
        raise HTTPException(
            status_code=400,
            detail="Expected 10 features (adult income)",
        )

    feature_row = pd.DataFrame([req.features], columns=FEATURE_NAMES)
    prediction = int(request.app.state.model.predict(feature_row)[0])
    label = "thu_nhap_cao" if prediction == 1 else "thu_nhap_thap"
    return {"prediction": prediction, "label": label}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
