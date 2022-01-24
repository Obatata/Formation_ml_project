import math

from prediction_model.config.config_parser import configutations
from prediction_model.feature_utils.features_encoding_data import\
    OrdinalMapper, YeoJhonsonTransformer


def test_ordinal_mapper(input_data):
    """
    Test l'encoder ordinal mapper
    sur la variable RestingECG
    """
    ordinal_mapper = OrdinalMapper(
            variables=configutations.modelDataConfig.RestingECG_vars,
            mapping=configutations.modelDataConfig.map_RestingECG
        )
    # verifier la valeur de la donnée avant de l'encoder
    assert input_data["RestingECG"].iat[3] == "ST"
    # transformer les données avec le mapper en utilisant fit_transform
    transformed_data = ordinal_mapper.fit_transform(input_data)
    # verifier la valeur de la donnée après l'encoding du mapper
    assert transformed_data["RestingECG"].iat[3] == 2


def test_yeo_jhonson_transformer(input_data):
    """
    test le transformer YeoJhonson
    sur la variable RestingBP
    """
    # verfier la valeur de la donnée avant transformation
    assert input_data["RestingBP"].iat[3] == 115
    # instancier le transformer
    yeo_jhonson_transformer = YeoJhonsonTransformer(
        variables=configutations.
        modelDataConfig.numerical_yeao_jhonson_transformation,
        lmbda_yeo_johnson=configutations.
        modelDataConfig.lmbda_yeo_johnson
    )
    # transformer la donnée
    transformed_data = yeo_jhonson_transformer.transform(input_data)
    # verfier la valeur attendue
    assert math.isclose(transformed_data["RestingBP"].iat[3], 0.079663, abs_tol=0.01)
