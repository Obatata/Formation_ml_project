from numpy import inexact

from prediction_model.config.config_parser import configutations
import pandas as pd
from prediction_model.feature_utils.data_and_pipeline_manager \
    import load_pipeline
from prediction_model.feature_utils.data_and_type_validator \
    import validate_batch_inference_input_data


serialized_pipeline_name = \
    f"{configutations.app_config.pipeline_serialized_file}.pkl"
heart_classification_pipeline = \
    load_pipeline(file_name=serialized_pipeline_name)

def batch_inference(input_data: pd.DataFrame) -> dict:
    """
    Make prediction in batch inference way
    with saved pipeline
    """
    validated_data, errors = \
        validate_batch_inference_input_data(input_data=input_data)

    resultats = {"predictions": None, "errors": errors}

    if not errors:
        classifications = heart_classification_pipeline.predict(
            X=validated_data[configutations.modelDataConfig.features]
        )
        resultats = {
            "classifications": [pred for pred in classifications],
            "errors": errors
        }
    return resultats
