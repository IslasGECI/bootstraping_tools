from bootstrapping_tools import AbstractSeriesBootstrapper, Bootstrap_from_time_series_parametrizer

import pytest

independent_variable = "Capturas"
bootstrap_number = 100
parametrizer = Bootstrap_from_time_series_parametrizer(
    blocks_length=1,
    column_name="CPUE",
    N=bootstrap_number,
    independent_variable=independent_variable,
)


def test_abstract_bootstrapp():
    class Dummy(AbstractSeriesBootstrapper):
        def __init__(self, bootstrapper_parametrizer):
            self.parameters_distribution = self.get_parameters_distribution()

        def get_parameters_distribution(self):
            pass

        def save_intervals(self, output_path):
            pass

    obtained = Dummy(parametrizer)
    assert obtained.parameters_distribution is None
    obtained.save_intervals("path")


@pytest.mark.xfail(strict=True)
def tests_abstract_method_error():
    class Dummy_without_save_intervals(AbstractSeriesBootstrapper):
        def __init__(self, bootstrapper_parametrizer):
            self.parameters_distribution = self.get_parameters_distribution()

        def get_parameters_distribution(self):
            pass

    Dummy_without_save_intervals(parametrizer)


@pytest.mark.xfail(strict=True)
def tests_abstract_method_error_parameters_distribution():
    class Dummy_without_parameters_distribution(AbstractSeriesBootstrapper):
        def __init__(self, bootstrapper_parametrizer):
            pass

        def save_intervals(self):
            pass

    Dummy_without_parameters_distribution(parametrizer)
