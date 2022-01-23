from prediction_model.config.config_parser import (
    lire_config_yml,
    parse_config_file
)


def test_parser_config_schema(dir_config):
    """
    test le schema du fihier config parsé
    vous pouvez voire le schema du fichier config parsé
    le schema des configs est deux parties :
        1. AppConfig
        2. MdelDataConfig

    dans le fichier :
    prediction_model/config/config_parser.py
    """
    cfg_yml = lire_config_yml(cfg_path=dir_config)
    configs = parse_config_file(parsed_config=cfg_yml)
    # assert the schema of the configs
    assert configs.app_config
    assert configs.modelDataConfig


def test_config_solver_logistic_regression(dir_config):
    """
    tester l'argument solver de la regression logistic
    """
    cfg_yml = lire_config_yml(cfg_path=dir_config)
    configs = parse_config_file(parsed_config=cfg_yml)
    assert configs.modelDataConfig.solver in configs.modelDataConfig.lr_solvers