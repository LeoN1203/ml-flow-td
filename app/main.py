from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Dict, List, Any, Optional
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
import logging
import os
import time
import requests
import random
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLflow Model Serving API with Canary Deployment",
    description="Web service to serve ML models from MLflow with canary deployment support",
    version="2.0.0"
)

# Global model storage for canary deployment
current_model = None
next_model = None
current_model_info = {
    "name": None,
    "version": None,
    "loaded_at": None,
    "uri": None
}
next_model_info = {
    "name": None,
    "version": None,
    "loaded_at": None,
    "uri": None
}

# Canary deployment configuration
canary_probability = float(os.getenv("CANARY_PROBABILITY", "0.1"))

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "StudentExamPredictor")
DEFAULT_MODEL_VERSION = os.getenv("DEFAULT_MODEL_VERSION", "latest")

# Force all MLflow operations to use container paths
os.environ['MLFLOW_ARTIFACT_URI'] = 'file:///mlflow/mlartifacts'
os.environ['MLFLOW_BACKEND_STORE_URI'] = 'file:///mlflow/mlruns'
os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT'] = 'file:///mlflow/mlartifacts'

# Pydantic models for API
class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "features": {
                    "study_hours": 6.0,
                    "attendance_rate": 0.95,
                    "previous_grade": 85.0,
                    "sleep_hours": 8.0,
                    "exercise_hours": 3.0
                }
            }
        }
    )
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prediction: Any
    model_info: Dict[str, Any]
    prediction_time: str
    model_used: str  # "current" or "next"

class ModelUpdateRequest(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "model_name": "StudentExamPredictor",
                "model_version": "2"
            }
        }
    )
    model_name: str
    model_version: Optional[str] = "latest"

class ModelUpdateResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    success: bool
    message: str
    previous_model: Dict[str, Any]
    new_model: Dict[str, Any]

class CanaryStatusResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    current_model: Dict[str, Any]
    next_model: Dict[str, Any]
    canary_probability: float
    models_identical: bool

class CanaryProbabilityRequest(BaseModel):
    probability: float


def wait_for_mlflow(max_retries=30, retry_delay=2):
    """Wait for MLflow to be available before starting"""
    mlflow_health_url = f"{MLFLOW_TRACKING_URI}/health"
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}: Checking MLflow availability at {mlflow_health_url}")
            response = requests.get(mlflow_health_url, timeout=5)
            if response.status_code == 200:
                logger.info("✅ MLflow is ready!")
                return True
        except Exception as e:
            logger.info(f"MLflow not ready yet: {str(e)}")
        if attempt < max_retries - 1:
            logger.info(f"Waiting {retry_delay} seconds before retry...")
            time.sleep(retry_delay)
    logger.warning("⚠️ MLflow not available after maximum retries - starting anyway")
    return False


def load_model_from_mlflow(model_name: str, model_version: str = "latest") -> tuple:
    """Load model from MLflow server with path resolution fix"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        if model_version.lower() == "latest":
            model_uri = f"models:/{model_name}/Latest"
        else:
            model_uri = f"models:/{model_name}/{model_version}"

        logger.info(f"Loading model from URI: {model_uri}")

        model = None

        try:
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info("✅ Model loaded using model registry URI")
        except Exception as registry_error:
            logger.warning(f"Model registry loading failed: {str(registry_error)}")

        if model is None:
            raise Exception("Model loading returned None")

        info = {
            "name": model_name,
            "version": model_version,
            "loaded_at": datetime.now().isoformat(),
            "uri": model_uri
        }
        logger.info(f"Successfully loaded model: {model_name} version {model_version}")
        return model, info

    except Exception as e:
        logger.error(f"Failed to load model {model_name} version {model_version}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Enhanced startup with canary deployment setup"""
    global current_model, next_model, current_model_info, next_model_info

    logger.info("🔍 Waiting for MLflow to be available...")
    wait_for_mlflow()

    try:
        logger.info("Loading default model for both current and next on startup...")
        # Load the same model for both current and next initially
        current_model, current_model_info = load_model_from_mlflow(DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION)
        next_model, next_model_info = load_model_from_mlflow(DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION)

        logger.info(f"Startup complete. Loaded model: {current_model_info}")
        logger.info(f"Canary probability set to: {canary_probability}")
    except Exception as e:
        logger.warning(f"Failed to load default model on startup: {str(e)}")
        logger.warning("Service started without a loaded model - this is OK if no model is registered yet")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "MLflow Model Serving API with Canary Deployment",
        "status": "healthy",
        "current_model": current_model_info,
        "next_model": next_model_info,
        "canary_probability": canary_probability,
        "mlflow_uri": MLFLOW_TRACKING_URI
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "current_model_loaded": current_model is not None,
        "next_model_loaded": next_model is not None,
        "current_model_info": current_model_info,
        "next_model_info": next_model_info,
        "canary_probability": canary_probability,
        "mlflow_uri": MLFLOW_TRACKING_URI,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/canary-status", response_model=CanaryStatusResponse)
async def get_canary_status():
    """Get canary deployment status"""
    models_identical = (
        current_model_info.get("name") == next_model_info.get("name") and
        current_model_info.get("version") == next_model_info.get("version")
    )
    return CanaryStatusResponse(
        current_model=current_model_info,
        next_model=next_model_info,
        canary_probability=canary_probability,
        models_identical=models_identical
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using canary deployment logic"""
    global current_model, next_model, current_model_info, next_model_info, canary_probability

    if current_model is None:
        raise HTTPException(status_code=503, detail="No current model is loaded")

    # Canary deployment logic: use next model with probability p
    use_next_model = random.random() < canary_probability and next_model is not None

    try:
        features_df = pd.DataFrame([request.features])
        logger.info(f"Making prediction with features: {request.features}")

        if use_next_model:
            model_to_use = next_model
            model_info_to_use = next_model_info
            model_used = "next"
            logger.info("Using NEXT model for prediction (canary)")
        else:
            model_to_use = current_model
            model_info_to_use = current_model_info
            model_used = "current"
            logger.info("Using CURRENT model for prediction")

        # Make prediction
        prediction = model_to_use.predict(features_df)

        # Handle different prediction formats
        if isinstance(prediction, np.ndarray):
            if len(prediction) == 1:
                prediction_result = prediction[0]
            else:
                prediction_result = prediction.tolist()
        else:
            prediction_result = prediction

        response = PredictionResponse(
            prediction=prediction_result,
            model_info=model_info_to_use,
            prediction_time=datetime.now().isoformat(),
            model_used=model_used
        )

        logger.info(f"Prediction successful using {model_used} model: {prediction_result}")
        return response

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/update-model", response_model=ModelUpdateResponse)
async def update_next_model(request: ModelUpdateRequest):
    """Update the NEXT model (canary model) with a new version from MLflow"""
    global next_model, next_model_info

    # Store previous model info
    previous_info = next_model_info.copy()

    try:
        # Load new model for next/canary
        new_model, new_info = load_model_from_mlflow(request.model_name, request.model_version)

        # Update next model and info
        next_model = new_model
        next_model_info = new_info

        logger.info(f"Successfully updated NEXT model to {request.model_name} version {request.model_version}")

        return ModelUpdateResponse(
            success=True,
            message=f"Successfully updated NEXT model to {request.model_name} version {request.model_version}",
            previous_model=previous_info,
            new_model=new_info
        )

    except Exception as e:
        logger.error(f"Next model update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Next model update failed: {str(e)}")


@app.post("/accept-next-model")
async def accept_next_model():
    """Accept the next model as current (promote canary to production)"""
    global current_model, next_model, current_model_info, next_model_info

    if next_model is None:
        raise HTTPException(status_code=400, detail="No next model is loaded to accept")

    try:
        previous_current = current_model_info.copy()

        current_model = next_model
        current_model_info = next_model_info.copy()

        # Keep next model the same (both current and next are now identical)
        # This maintains the canary setup for future updates

        logger.info(f"Successfully promoted next model to current: {current_model_info}")

        return {
            "success": True,
            "message": "Next model successfully accepted as current",
            "previous_current_model": previous_current,
            "new_current_model": current_model_info,
            "next_model": next_model_info
        }

    except Exception as e:
        logger.error(f"Failed to accept next model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to accept next model: {str(e)}")


@app.get("/model-info")
async def get_model_info():
    """Get information about both current and next models"""
    return {
        "current_model": current_model_info,
        "next_model": next_model_info,
        "canary_probability": canary_probability,
        "current_model_loaded": current_model is not None,
        "next_model_loaded": next_model is not None,
        "available_models": await get_available_models()
    }

async def get_available_models():
    """Get list of available models from MLflow"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        registered_models = client.search_registered_models()

        models_info = []
        for model in registered_models:
            versions = client.search_model_versions(f"name='{model.name}'")
            latest_versions = [v.version for v in versions]
            models_info.append({
                "name": model.name,
                "versions": latest_versions,
                "description": model.description
            })

        return models_info
    except Exception as e:
        logger.error(f"Failed to get available models: {str(e)}")
        return []
