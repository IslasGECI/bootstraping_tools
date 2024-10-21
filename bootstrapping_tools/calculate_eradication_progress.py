from bootstrapping_tools.calculate_growth_rates import AbstractSeriesBootstrapper


class ProgressBootstrapper(AbstractSeriesBootstrapper):
    def __init__(self, bootstrapper_parametrizer):
        bootstrapper_parametrizer.parameters["dataframe"].rename(
            columns={"Capturas": "Temporada", "Esfuerzo": "CPUE"}, inplace=True
        )
        super().__init__(bootstrapper_parametrizer)
