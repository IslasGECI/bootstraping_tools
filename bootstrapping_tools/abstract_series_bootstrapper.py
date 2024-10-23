from abc import ABC, abstractmethod

from bootstrapping_tools.bootstrapping import (
    calculate_intervals_from_p_values_and_alpha,
    calculate_p_values,
    generate_latex_interval_string,
)


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

    def get_parameters_dictionary(self):
        json_dict = {
            "intervals": list(self.intervals),
            "main_parameter_latex_interval": self.lambda_latex_interval,
            "p-values": self.p_values,
            "bootstrap_intermediate_distribution": self.get_parameters_inside_confidence_interval(),
        }
        return json_dict

    @abstractmethod
    def save_intervals(self):
        pass

    def get_parameters_inside_confidence_interval(self):
        return [
            parameters_tuple
            for parameters_tuple in self.parameters_distribution
            if (parameters_tuple[0] > self.intervals[0][0])
            and (parameters_tuple[0] < self.intervals[2][0])
        ]
