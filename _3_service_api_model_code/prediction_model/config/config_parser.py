# native python libraries
from pathlib import Path
from typing import Dict, List
from pydantic import BaseModel
from strictyaml import load, YAML


ROOT_PREDICTION_MODEL = Path(__file__).parent.parent
ROOT = ROOT_PREDICTION_MODEL.parent
CONFIG_FILE_PATH = ROOT / "config.yml"
DATASETS_PATH = ROOT_PREDICTION_MODEL / "datasets"
SERIALIZED_MODELS = ROOT_PREDICTION_MODEL / "serialized_models"


class AppConfig(BaseModel):
    """
    Les configs de haut niveau d'application
    Le fichier config.yml contient ces confgis
    """

    training_data_file: str
    test_data_file: str
    pipeline_serialized_file: str


class MdelDataConfig(BaseModel):
    """
    Les configs relatives à model data, transformation,
    imputation, mapping et oneHotEncofing
    Le fichier config.yml reporte ces configs
    """

    target: str
    features: List[str]
    test_size: float
    random_state: int
    lmbda_yeo_johnson: float
    C: float
    lr_penality: str
    solver: str
    lr_solvers: List[str]
    categorical_variables_with_na_missing: List[str]
    categorical_variables_with_na_frequent: List[str]
    numerical_variables_with_na: List[str]
    numerical_yeao_jhonson_transformation: List[str]
    variables_one_hot_encoding: List[str]
    RestingECG_vars: List[str]
    Sex_vars: List[str]
    ExerciseAngina_vars: List[str]
    map_RestingECG: Dict[str, int]
    map_binary_RestingECG: Dict[str, int]
    map_Sex: Dict[str, int]
    map_ExerciseAngina: Dict[str, int]


class AllConfig(BaseModel):
    """
    Schéma global des configuration du projet
    en deux groupes :
    -- les configs d'appli (AppConfig)
    -- les configs du modèle data (MOdelDataConfig)
    """
    app_config: AppConfig
    modelDataConfig: MdelDataConfig


def check_config_file() -> Path:
    """
    Cette fonction doit vérifier si le fihcier config yml existe
    :return: soit le nom du fichier yml soit une Exception
    """
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception("Le fichier config {} "
                    "n'éxiste pas ! ".format(CONFIG_FILE_PATH))


def lire_config_yml(cfg_path: Path = None) -> YAML:
    """
    check confgi path
    load config path if exist
    :return: parsed config file
    """
    cfg_path = check_config_file()
    if cfg_path:
        with open(cfg_path, "r") as config_file:
            parsed_file = load(config_file.read())
            return parsed_file
    raise Exception("file {} doest not existe".format(cfg_path))


def parse_config_file(parsed_config: YAML = None) -> AllConfig:
    """
    read and get config file
    get configs from the config file

    :return: configs (api_schemas and modelData)
    """
    parsed_config = lire_config_yml()
    configs_project = AllConfig(
        app_config=AppConfig(**parsed_config.data),
        modelDataConfig=MdelDataConfig(**parsed_config.data)
        )
    return configs_project


configutations = parse_config_file()