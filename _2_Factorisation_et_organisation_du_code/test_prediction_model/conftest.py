import pytest
from pathlib import  Path

from prediction_model.config.config_parser import configutations
from prediction_model.feature_utils.data_and_pipeline_manager import\
    load_dataset, load_pipeline


@pytest.fixture()
def input_data():
    """
    Cette fonction est en charge d'importer les données de test
    Ces données tests seront partagé sur toutes les fonctions
    tests qu'on codé pour tester nos fonctionalités, configs et
    performance du model.
    @pytest.fixture() ==> decorateur utilité pour le partage
    des données importées

    lien documentation :
    https://docs.pytest.org/en/6.2.x/fixture.html
    """
    return load_dataset(
        file_name_data=configutations.app_config.test_data_file)

@pytest.fixture()
def pipeline_heart():
    """
    Dans cette fonction on instancie notre pipeline
    pour la aprtager sur les fonction test
    de performance de notre modèle de classification
    """
    serialized_pipeline_name = \
        f"{configutations.app_config.pipeline_serialized_file}.pkl"
    return load_pipeline(file_name=serialized_pipeline_name)

@pytest.fixture()
def dir_config():
    return Path(__file__).parent.parent / "config.yml"
