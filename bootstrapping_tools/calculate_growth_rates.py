from bootstrapping_tools.bootstrapping import (
    bootstrap_from_time_series,
    get_bootstrap_deltas,
    lambda_calculator,
    power_law,
)

from bootstrapping_tools.abstract_series_bootstrapper import AbstractSeriesBootstrapper

import json


class Bootstrap_from_time_series_parametrizer:
    def __init__(
        self,
        blocks_length=3,
        N=2000,
        column_name="Maxima_cantidad_nidos",
        independent_variable="Temporada",
        alpha=0.05,
    ):
        self.parameters = dict(
            dataframe=None,
            column_name=column_name,
            N=N,
            return_distribution=True,
            blocks_length=blocks_length,
            alpha=alpha,
        )
        self.independent_variable = independent_variable

    def set_data(self, data):
        data["Temporada"] = data[self.independent_variable]
        self.parameters["dataframe"] = data


def fit_population_model(seasons_series, data_series):
    parameters = lambda_calculator(seasons_series, data_series)
    model = power_law(
        seasons_series - seasons_series.iloc[0],
        parameters[0],
        parameters[1],
    )
    return model


class LambdasBootstrapper(AbstractSeriesBootstrapper):
    def __init__(self, bootstrap_parametrizer):
        self.bootstrap_config = bootstrap_parametrizer.parameters
        self.data_series = self.bootstrap_config["dataframe"][self.bootstrap_config["column_name"]]
        self.season_series = self.bootstrap_config["dataframe"]["Temporada"]
        self.parameters_distribution, _ = self.get_parameters_distribution()

    def get_parameters_distribution(self):
        lambdas_n0_distribution, intervals = bootstrap_from_time_series(**self.bootstrap_config)
        return lambdas_n0_distribution, intervals

    def get_distribution(self):
        return self.parameters_distribution

    def get_inferior_central_and_superior_limit(self):
        inferior_limit, central, superior_limit = get_bootstrap_deltas(
            self.interval_lambdas, **{"decimals": 2}
        )
        return inferior_limit, central, superior_limit

    def fit_population_model(self):
        model = fit_population_model(self.season_series, self.data_series)
        return model

    def save_intervals(self, output_path):
        json_dict = self.get_parameters_dictionary()
        json_dict["lambda_latex_interval"] = json_dict.pop("main_parameter_latex_interval")
        with open(output_path, "w") as file:
            json.dump(json_dict, file)
