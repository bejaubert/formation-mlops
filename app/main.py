"""
Main file for the API.
"""
import os
from contextlib import asynccontextmanager
from typing import List, Dict
from fastapi import FastAPI
from pydantic import BaseModel

from app.utils import (
    get_model,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager for managing the lifespan of the API.
    This context manager is used to load the ML model and other resources
    when the API starts and clean them up when the API stops.
    Args:
        app (FastAPI): The FastAPI application.
    """
    global model

    model_name: str = os.getenv("MLFLOW_MODEL_NAME")
    model_version: str = os.getenv("MLFLOW_MODEL_VERSION")
    # Load the ML model
    model = get_model(model_name, model_version)
    yield


class ActivityDescriptions(BaseModel):
    """
    Pydantic BaseModel for representing the input data for the API.
    This BaseModel defines the structure of the input data required
    for the API's "/predict-batch" endpoint.

    Attributes:
        text_descriptions (List[str]): The text descriptions.
    """

    text_descriptions: List[str]

    class Config:
        schema_extra = {
            "example": {
                "text_description": [
                    (
                        "LOUEUR MEUBLE NON PROFESSIONNEL EN RESIDENCE DE "
                        "SERVICES (CODE APE 6820A Location de logements)"
                    )
                ]
            }
        }


app = FastAPI(
    lifespan=lifespan,
    title="NACE classifier",
    description="Classifier for firm activity descriptions",
    version="0.0.1",
)


@app.get("/", tags=["Welcome"])
def show_welcome_page():
    """
    Show welcome page with current model name and version.
    """
    model_name: str = os.getenv("MLFLOW_MODEL_NAME")
    model_version: str = os.getenv("MLFLOW_MODEL_VERSION")
    return {
        "message": "NACE classifier",
        "model_name": f"{model_name}",
        "model_version": f"{model_version}",
    }


@app.get("/predict", tags=["Predict"])
async def predict(
    description: str,
    nb_echoes_max: int = 5,
) -> Dict:
    """
    Predict NACE code.
    """
    # Conversion de l'input en liste simple
    input_data = [description]

    predictions = model.predict(
        input_data,  # Liste de strings attendue par MLflow
        params={"k": nb_echoes_max}  # Passage correct des paramètres
    )

    return predictions[0]  # Même format de retour qu'avant
