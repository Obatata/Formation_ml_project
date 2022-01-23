from prediction_model.config.config_parser import configutations


def test_accuracy_model(input_data, pipeline_heart):
    """
    tester la performance du modèle
    on se fixe uns seuil de 0.5
    """
    # scorer la pipline du conftest
    score_accuracy = pipeline_heart.score(
        input_data[configutations.modelDataConfig.features],
        input_data[configutations.modelDataConfig.target])
    # verifier que l'accuracy du modèle est > 50 %
    assert score_accuracy > 0.5
