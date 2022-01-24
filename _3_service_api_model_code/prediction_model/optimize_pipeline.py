import numpy as np
from prediction_model.config.config_parser import configutations
from prediction_model.pipeline import heart_disease_pipe
from prediction_model.feature_utils.data_and_pipeline_manager import (
    load_dataset,
    save_pipeline,
    load_pipeline,
)
from prediction_model.feature_utils import\
    features_encoding_data as custoEncoder
from sklearn.model_selection import GridSearchCV, train_test_split


# on va lire la pipeline serialisé lors de l'entrainement du modèle
serialized_pipeline_name = \
    f"{configutations.app_config.pipeline_serialized_file}.pkl"
heart_classification_pipeline_train = \
    load_pipeline(file_name=serialized_pipeline_name)


def run_grid_search_cv() -> None:
    """
    Danc cette fonction nous allons optimiser
    la pipeline en fonction des paramètres
    de chaque step.
    En réalité on mettera le focus sur le step du
    modèle de la régression logistique.

    Si l'optimisation de la pipeline donne une meilleure
    performance (accuracy par ex) alors on serialise le
    modèle et la pipeline du précédent.
    """
    # lire les données d'entrainement via les configuration yml parsés
    data = load_dataset(file_name_data=
                        configutations.app_config.training_data_file)
    # On utilisera le train_test_split random de sklearn pour
    # répartir les données en train/test data
    X_train, X_test, y_train, y_test = train_test_split(
        # les variables features
        data[configutations.modelDataConfig.features],
        # la target
        data[configutations.modelDataConfig.target],
        test_size=configutations.modelDataConfig.test_size,
        # le paramètre seed est dans le config yaml
        # pour assurer la reproductibilité à la phase de
        # déploiement
        random_state=configutations.modelDataConfig.random_state
    )
    # on veut optimiser la pipeline on fonction de l'encoding des données
    # encoding 1 :
    encoder_1 = custoEncoder.OrdinalMapper(
        variables=configutations.modelDataConfig.RestingECG_vars,
        mapping=configutations.modelDataConfig.map_RestingECG)
    encoder_2 = custoEncoder.OrdinalMapper(
                  variables=configutations.modelDataConfig.RestingECG_vars,
                  mapping=configutations.modelDataConfig.map_binary_RestingECG)

    # Les paramètres de la regression
    parametres = {
        "ordinal_mapper_RestingECG" : [encoder_1, encoder_2],
        "logistic_regression_binaire__C" : np.logspace(-4, 4, 20),
        "logistic_regression_binaire__penalty" : ["l2", "none"],
        "logistic_regression_binaire__solver" : ["newton-cg", "lbfgs", "sag"]
    }

    # Creation de l'optimisation pippeline
    grid_opt = GridSearchCV(
        heart_classification_pipeline_train,
        parametres,cv=3,
    )

    # Optimisation de la pipeline
    grid_opt.fit(X_train, y_train)

    # restitution des résultats
    print("training accuracy is {} testing accuracy is {}".format(
        grid_opt.score(X_train, y_train),
        grid_opt.score(X_test, y_test)
        )
    )
    # Access the best set of parameters
    best_params = grid_opt.best_params_
    print("best params : ", best_params)
    import pandas as pd
    result_df = pd.DataFrame.from_dict(grid_opt.cv_results_, orient='columns')
    print()
    # comparer les résultat entre les deux pipelnes
    # on prend le score de la pipeline qui résulte de
    # train_pipeline et le compare avec le résultat
    # de la l'optimisation avec grid
    score_pipeline_train = heart_classification_pipeline_train.score(X_test, y_test)
    score_pipeline_grid_opt = grid_opt.score(X_test, y_test)
    print("old accuracy : {}\n  new_accuracy : {}".
          format(score_pipeline_train, score_pipeline_grid_opt))
    if score_pipeline_grid_opt > score_pipeline_train:
        opt_pipeline = grid_opt.best_estimator_
        save_pipeline(pipeline_to_serialize=opt_pipeline)


if __name__ == "__main__":
    run_grid_search_cv()