from feature_engine.imputation import (
    CategoricalImputer,
    AddMissingIndicator,
    MeanMedianImputer,
)
from prediction_model.feature_utils import features_encoding_data as custoEncoder
from sklearn.pipeline import Pipeline
from prediction_model.config.config_parser import configutations
from feature_engine.encoding import OneHotEncoder
from sklearn.linear_model import LogisticRegression

heart_disease_pipe = Pipeline(
    steps=[
            # ============= Imputation de variables manquantes ================
            # ------------------------------------------------------------------

            # Step 1 : Imputation de la valeur "missing" dans dans les vals nan
            # des variables qualitatives
            (
                "missing_imputer",
                CategoricalImputer(
                    imputation_method="missing",
                    variables=configutations.
                    modelDataConfig.categorical_variables_with_na_missing
                ),
            ),
            # Step 2 : Imputationtion de la valeur la plus fréquente
            # dans les vals nan des variables qualitatives
            (
                "frequent_imputer",
                CategoricalImputer(
                    imputation_method="frequent",
                    variables=configutations.
                    modelDataConfig.categorical_variables_with_na_frequent
                ),
            ),
            # Step 3 : Ajouter un indicateur de nan values pour
            # les variables quantitatives
            (
                "missing_indicator",
                AddMissingIndicator(
                    variables=configutations.
                    modelDataConfig.numerical_variables_with_na
                ),
            ),
            # Step 4 : Imputer la valeur moyenne sur les nan valuues pour
            # las variables quantitatives
            (
                "mean_imputer",
                MeanMedianImputer(
                    imputation_method="mean",
                    variables=configutations.
                    modelDataConfig.numerical_variables_with_na
                ),
            ),

            # ============= Transformation des données quantitatives ===========
            # ------------------------------------------------------------------

            #  Step 5 : transformer la liste des
            #  variables quantitatives asymétriques
            (
              "yeo_johnson_transformer",
              custoEncoder.YeoJhonsonTransformer(
                  variables=configutations.
                  modelDataConfig.numerical_yeao_jhonson_transformation,
                  lmbda_yeo_johnson=configutations.
                  modelDataConfig.lmbda_yeo_johnson
              ),
            ),
            # ============= Mapping des données qualitatives  ==================
            # ------------------------------------------------------------------

            # Step 6 : mapping des variables qualitatives ordinales
            (
              "ordinal_mapper_RestingECG",
              custoEncoder.OrdinalMapper(
                  variables=configutations.modelDataConfig.RestingECG_vars,
                  mapping=configutations.modelDataConfig.map_RestingECG
              ),
            ),

            (
              "ordinal_mapper_sex",
              custoEncoder.OrdinalMapper(
                  variables=configutations.modelDataConfig.Sex_vars,
                  mapping=configutations.modelDataConfig.map_Sex
              ),
            ),

            (
              "ordinal_mapper_exerciseAngina",
              custoEncoder.OrdinalMapper(
                  variables=configutations.modelDataConfig.ExerciseAngina_vars,
                  mapping=configutations.modelDataConfig.map_ExerciseAngina
              ),
            ),

            # Step 7 : mapping des variables qualitatives ordinales
            (
              "one-hot-encoder_mapper",
              OneHotEncoder(
                  variables=configutations.
                      modelDataConfig.variables_one_hot_encoding,
              ),
            ),
            # ============= Classification par regression logistique =========
            # ------------------------------------------------------------------
            # Step 8 : Régression logistique
            (
               "logistic_regression_binaire",
                LogisticRegression(
                    C=configutations.modelDataConfig.C,
                    solver=configutations.modelDataConfig.solver
                )
            )
    ]
)
