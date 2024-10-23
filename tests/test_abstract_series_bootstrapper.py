from bootstrapping_tools import AbstractSeriesBootstrapper, Bootstrap_from_time_series_parametrizer


def test_abstract_bootstrapp():
    AbstractSeriesBootstrapper.__abstractmethods__ = set()

    class Dummy(AbstractSeriesBootstrapper):
        def __init__(self, bootstrapper_parametrizer):
            pass

    independent_variable = "Capturas"
    bootstrap_number = 100
    parametrizer = Bootstrap_from_time_series_parametrizer(
        blocks_length=1,
        column_name="CPUE",
        N=bootstrap_number,
        independent_variable=independent_variable,
    )
    obtained = Dummy(parametrizer)
    assert obtained.parameters_distribution is None
