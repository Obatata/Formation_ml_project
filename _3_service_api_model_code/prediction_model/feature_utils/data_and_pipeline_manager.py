from pathlib import Path
import pandas as pd
from prediction_model.config.config_parser import DATASETS_PATH, SERIALIZED_MODELS
from sklearn.pipeline import Pipeline
from prediction_model.config.config_parser import configutations, SERIALIZED_MODELS
import joblib


def load_dataset(file_name_data: str) -> pd.DataFrame:
    """
    Lire les données d'entraineement ou de test
    qui sont localisés dans le répértoire "datasets"

    rmq : l'objectif de cette fonction est d'isoler la lecture et
        les potentiels traitement qu'on peut ajouter avcant la
        phase d'entrainement du model !
        par exemple modifier la distribution de la variables target
        ou le nom de certaines colonnes avant d'envoyer les données
        à la pipeline

    :param file_name_data:
    :return: data_frame
    """

    data_frame = pd.read_csv(Path(f"{DATASETS_PATH}/{file_name_data}"),
                             sep=",")
    return data_frame

def save_pipeline(pipeline_to_serialize: Pipeline) -> None:
    """
    Serialize the pipeline
    """

    save_file_name = f"{configutations.app_config.pipeline_serialized_file}" \
                     f".pkl"
    save_path = SERIALIZED_MODELS / save_file_name
    # save the file with joblib
    joblib.dump(pipeline_to_serialize, save_path)


def load_pipeline(file_name: str) -> Pipeline:
    """
    Load serialized pipeline
    return: trained_pipeline
    """
    file_path = SERIALIZED_MODELS / file_name
    trained_pipeline = joblib.load(file_path)
    return trained_pipeline


