from prediction_model.config.config_parser import configutations
from prediction_model.feature_utils.features_encoding_data import\
    OrdinalMapper, YeoJhonsonTransformer
import pytest

from prediction_model.config.config_parser import configutations
from prediction_model.feature_utils.data_and_pipeline_manager import\
    load_dataset, load_pipeline
import math
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


if __name__ == "__main__":
    serialized_pipeline_name = \
        f"{configutations.app_config.pipeline_serialized_file}.pkl"
    heart_classification_pipeline = \
        load_pipeline(file_name=serialized_pipeline_name)

    print(heart_classification_pipeline)
    data = input_data()
    y_prediction = heart_classification_pipeline.predict(data[configutations.modelDataConfig.features])
    print(y_prediction)
    score = heart_classification_pipeline.score(data[configutations.modelDataConfig.features], data[configutations.modelDataConfig.target])
    print("score : ", score)