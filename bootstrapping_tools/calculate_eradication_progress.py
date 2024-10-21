from bootstrapping_tools.calculate_growth_rates import AbstractSeriesBootstrapper

import numpy as np


class ProgressBootstrapper(AbstractSeriesBootstrapper):
    def __init__(self, bootstrapper_parametrizer):
        bootstrapper_parametrizer.parameters["dataframe"]["CPUE"] = 1
        super().__init__(bootstrapper_parametrizer)
        self.data_series = self.add_cpue()

    def add_cpue(self):
        data = self.parameters["dataframe"]
        data["CPUE"] = data.Capturas / data.Esfuerzo
        return data

    @property
    def parameters_distribution(self):
        rng = np.random.default_rng(42)
        distribution = []
        distribution_size = 0
        captures = self.data_series.Capturas.sum()
        while self.parameters["N"] > distribution_size:
            sample = resample_eradication_data(self.data_series, rng)
            parameters = fit_ramsey_plot(sample)
            is_valid = -parameters[1] / parameters[0] > captures
            if is_valid:
                distribution.append(parameters)
            distribution_size = len(distribution)
        return distribution


def fit_ramsey_plot(data):
    assert len(data["Cumulative_captures"].unique()) > 1, "It can not fit Ramsey plot"
    fit = np.polynomial.polynomial.Polynomial.fit(data["Cumulative_captures"], data["CPUE"], deg=1)
    intercept_and_slope = fit.convert().coef
    idx = [1, 0]
    slope_and_intercept = intercept_and_slope[idx]
    return slope_and_intercept


def resample_eradication_data(data, rng):
    resampled_data = data.sample(replace=True, frac=1, random_state=rng)
    sorted_data = resampled_data.sort_index()
    sorted_data["Cumulative_captures"] = sorted_data.Capturas.cumsum()
    return sorted_data[["CPUE", "Cumulative_captures"]]
