from typing import Any
import pandas as pd
from api_schemas import batch_inference_schema
from prediction_model.inference_pipeline import batch_inference
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder


app = FastAPI()
@app.post("/predict_batch",
          response_model=batch_inference_schema.PredictionResults)
def batch_inference_prediction(
        input_data: batch_inference_schema.MultipleHeartDiseaseInputs
    ) -> Any:
    """
    Réaliser une prediction de lot (en batch)
    :param input_data:
    :return:
    """
    input_data = pd.DataFrame(jsonable_encoder(input_data.inputs))