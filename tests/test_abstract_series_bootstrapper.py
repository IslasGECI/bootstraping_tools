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
            self.bootstrap_config = bootstrapper_parametrizer.parameters

        def get_parameters_distribution(self):
            return [[1], [3], [5]]

        def save_intervals(self, output_path):
            pass

    obtained = Dummy(parametrizer)
    assert isinstance(obtained.parameters_distribution, list)
    obtained.save_intervals("path")
    assert isinstance(obtained.lambda_latex_interval, str)
    assert isinstance(obtained.get_parameters_dictionary(), dict)
    assert isinstance(obtained.get_parameters_inside_confidence_interval(), list)


@pytest.mark.xfail(strict=True)
def tests_abstract_method_error():
    class Dummy_without_save_intervals(AbstractSeriesBootstrapper):
        def __init__(self, bootstrapper_parametrizer):
            self.parameters_distribution = self.get_parameters_distribution()

        def get_parameters_distribution(self):
            pass

    obtained = Dummy_without_save_intervals(parametrizer)
    obtained.interval_lambdas is None


@pytest.mark.xfail(strict=True)
def tests_abstract_method_error_parameters_distribution():
    class Dummy_without_parameters_distribution(AbstractSeriesBootstrapper):
        def __init__(self, bootstrapper_parametrizer):
            pass

        def save_intervals(self):
            pass

    Dummy_without_parameters_distribution(parametrizer)
