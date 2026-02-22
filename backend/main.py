import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from contextlib import asynccontextmanager
from backend.routers.liveness import router
from backend.services.liveness_services import load_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — load models once
    print("Loading models...")
    load_models()
    print("Models loaded successfully")
    yield
    # Shutdown — nothing to clean up for now
    print("Shutting down...")


app = FastAPI(
    title="Liveness Detection API",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)


@app.get("/livenessCheck")
def liveness_check():
    return {"status": "ok"}
