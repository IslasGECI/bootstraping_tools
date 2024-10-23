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
