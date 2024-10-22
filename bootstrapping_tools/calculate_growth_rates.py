from bootstrapping_tools.bootstrapping import (
    bootstrap_from_time_series,
    calculate_intervals_from_p_values_and_alpha,
    calculate_p_values,
    generate_latex_interval_string,
    get_bootstrap_deltas,
    lambda_calculator,
    power_law,
)

import json
import numpy as np
from abc import ABC, abstractmethod


def calculate_seasons_intervals(seasons):
    years = []
    first_index = 0
    for index in np.where(np.diff(seasons) != 1)[0]:
        if seasons[first_index] == seasons[index]:
            years.append(f"{seasons[index]}")
        else:
            years.append(f"{seasons[first_index]}-{seasons[index]}")
        first_index = index + 1
    years.append(f"{seasons[first_index]}-{seasons[-1]}")
    return years


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


class AbstractSeriesBootstrapper(ABC):

    @abstractmethod
    def get_parameters_distribution(self):
        pass

    @property
    def interval_lambdas(self):
        return [interval[0] for interval in self.intervals]

    @property
    def p_values(self):
        lambdas = [lambdas_n0[0] for lambdas_n0 in self.parameters_distribution]
        p_value_mayor, p_value_menor = calculate_p_values(lambdas)
        p_values = (p_value_mayor, p_value_menor)
        return p_values

    @property
    def intervals(self):
        intervals = calculate_intervals_from_p_values_and_alpha(
            self.parameters_distribution, self.p_values, self.bootstrap_config["alpha"]
        )
        return intervals

    @property
    def lambda_latex_interval(self):
        lambda_latex_string = generate_latex_interval_string(
            self.interval_lambdas, deltas=False, **{"decimals": 2}
        )
        return lambda_latex_string

    def generate_season_interval(self):
        return "({}-{})".format(
            self.season_series.min(axis=0),
            self.season_series.max(axis=0),
        )

    def get_monitored_seasons(self):
        monitored_seasons = np.sort(self.season_series.astype(int).unique())
        if len(monitored_seasons) == 1:
            return f"{monitored_seasons[0]}"
        else:
            seasons_intervals = calculate_seasons_intervals(monitored_seasons)
            return ",".join(seasons_intervals)

    def save_intervals(self, output_path):
        json_dict = {
            "intervals": list(self.intervals),
            "lambda_latex_interval": self.lambda_latex_interval,
            "p-values": self.p_values,
            "bootstrap_intermediate_distribution": self.get_parameters_inside_confidence_interval(),
        }
        with open(output_path, "w") as file:
            json.dump(json_dict, file)

    def get_parameters_inside_confidence_interval(self):
        return [
            parameters_tuple
            for parameters_tuple in self.parameters_distribution
            if (parameters_tuple[0] > self.intervals[0][0])
            and (parameters_tuple[0] < self.intervals[2][0])
        ]


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
