from typing import Tuple, Optional, List
from pydantic import BaseModel, ValidationError
import pandas as pd
import numpy as np
from prediction_model.config.config_parser import configutations


def drop_na_inputs(inpu_data: pd.DataFrame) -> pd.DataFrame:
    """
    supprimer toutes les lignes contenant les nan values
    """
    # Il faut extraire les colonnes qui n'ont pas été traité en mode
    # nan value lors de l'entrainement du modèle.
    validated_data = inpu_data.copy()
    inference_var_with_na = [
        var for var in configutations.modelDataConfig.features
        if var not in
        configutations.modelDataConfig.categorical_variables_with_na_missing
        + configutations.modelDataConfig.categorical_variables_with_na_missing
        + configutations.modelDataConfig.numerical_variables_with_na
        and validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=inference_var_with_na, inplace=True)

    return validated_data


def validate_batch_inference_input_data(
        input_data: pd.DataFrame
) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Cette fonction est responsable de traiter les données d'inference
    lors du déploiement
    """
    # On s'assure que les données contiennet uniquement les colonnes
    # requises par le modèle entrainé
    validated_data = input_data[configutations.modelDataConfig.features].copy()

    # Lors du dépoliement du modèle en production
    # certaines données peuvent contenir des nan values
    validated_data = drop_na_inputs(inpu_data=validated_data)
    errors = None
    print("my data : \n", validated_data.replace({np.nan: None}).to_dict(orient="records"))
    try:
        # remplacer les nan values du dataframe par None
        # pour réaliser la validation via pydantic
        BatchDataInputsSchema(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


# Pydantic schemas
class RealTimeDataInputsSchema(BaseModel):
    Age: Optional[int]
    Sex: Optional[str]
    ChestPainType: Optional[str]
    RestingBP: Optional[int]
    Cholesterol: Optional[float]
    FastingBS: Optional[int]
    RestingECG: Optional[str]
    MaxHR: Optional[float]
    ExerciseAngina: Optional[str]
    Oldpeak: Optional[float]
    ST_Slope: Optional[str]


class BatchDataInputsSchema(BaseModel):
    inputs: List[RealTimeDataInputsSchema]
