from typing import Optional, Any, List
from pydantic import BaseModel
from prediction_model.feature_utils.data_and_type_validator import BatchDataInputsSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions:Optional[List[int]]


class MultipleHeartDiseaseInputs(BaseModel):
    inputs: List[BatchDataInputsSchema]
