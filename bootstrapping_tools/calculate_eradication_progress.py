from bootstrapping_tools.calculate_growth_rates import AbstractSeriesBootstrapper


class ProgressBootstrapper(AbstractSeriesBootstrapper):
    def __init__(self, bootstrapper_parametrizer):
        bootstrapper_parametrizer.parameters["dataframe"]["CPUE"] = 1
        super().__init__(bootstrapper_parametrizer)
        self.data_series = self.add_cpue()

    def add_cpue(self):
        data = self.parameters["dataframe"]
        data["CPUE"] = data.Capturas / data.Esfuerzo
        return data
