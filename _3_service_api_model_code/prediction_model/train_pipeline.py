from prediction_model.config.config_parser import configutations
from prediction_model.pipeline import heart_disease_pipe
from prediction_model.feature_utils.data_and_pipeline_manager import (
    load_dataset,
    save_pipeline
)
from sklearn.model_selection import train_test_split


def run_training_pipeline() -> None:
    """
    Nous allons entrainer la pipeline
    (model + transformers) sur l'ensemble des données
    d'entrainemeent.

    rmq : dans cette fonction aucune optimisation
          du modèle de classification n'est réalisée.

    :return: None
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

    # fit the model
    heart_disease_pipe.fit(X_train, y_train)

    # serialize the trained model
    save_pipeline(pipeline_to_serialize=heart_disease_pipe)

if __name__ == "__main__":
    run_training_pipeline()
