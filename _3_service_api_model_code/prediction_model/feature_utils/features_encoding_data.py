from typing import List, Dict
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from scipy import stats


class YeoJhonsonTransformer(BaseEstimator, TransformerMixin):
    """
    La transformation de YeoJhonsonTransformer doit suivre
    la pipeline sklearn
    """

    def __init__(self, variables: List[str], lmbda_yeo_johnson: float) -> None:
        # check type of variables structure
        if not isinstance(variables, list):
            raise ValueError("Attention les "
                             "variables ne sont pas de type list")

        self.variables = variables
        self.lmbda_yeo_johnson = lmbda_yeo_johnson

    def fit(self, x_data: pd.DataFrame, y_data: pd.Series):
        """
        Il est obligatoire d'ajouter la méthode fit
        pour suivre le shéma de la pipeline sklearn
        """
        return self

    def transform(self, x_data: pd.DataFrame) -> pd.DataFrame:
        """
        transformer chaque varaible en utilisant la transformation
        de yeo-jhonson
        Cette transformation est utilisé pour les variables
         asymitriques

         return:x_data
        """
        x_data = x_data.copy()
        for var in self.variables:
            x_data[var] = stats.yeojohnson(x_data[var],
                                           lmbda=self.lmbda_yeo_johnson)

        return x_data


class OrdinalMapper(BaseEstimator, TransformerMixin):
    """
    Le mapping des variables quaalitatives ordinale
    """

    def __init__(self, variables: List[str], mapping: Dict) -> None:
        if not isinstance(variables, list):
            raise ValueError("Attention les "
                             "variables ne sont pas de type list")

        self.variables = variables
        self.mapping = mapping

    def fit(self, x_data: pd.DataFrame, y_data: pd.Series = None):
        """
        Il est obligatoire d'ajouter la méthode fit
        pour suivre le shéma de la pipeline sklearn
        """
        return self

    def transform(self, x_data: pd.DataFrame) -> pd.DataFrame:
        """
        Dans cette fonction on applique la custom mapping
        qu'on a définit dans le config.yml

        return : x
        """
        # créer une copie pour ne pas modifier le contenu du
        # dataframe initial
        x_data = x_data.copy()
        # appliquer la mapping sur le dataframe
        for var in self.variables:
            x_data[var] = x_data[var].map(self.mapping)

        return x_data

