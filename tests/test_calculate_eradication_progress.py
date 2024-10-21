import pandas as pd


from bootstrapping_tools import Bootstrap_from_time_series_parametrizer, ProgressBootstrapper

raw_data = pd.DataFrame(
    {
        "Esfuerzo": [1 / 19.5, 2 / 19, 3 / 18.5, 4 / 18, 5 / 17.5, 6 / 17],
        "Capturas": [1, 2, 3, 4, 5, 6],
    }
)


def test_ProgressBootstrapper():
    independent_variable = "Capturas"
    parametrizer = Bootstrap_from_time_series_parametrizer(
        blocks_length=1, column_name="CPUE", N=100, independent_variable=independent_variable
    )
    parametrizer.set_data(raw_data)
    bootstrapper = ProgressBootstrapper(parametrizer)

    obtained_cpue = bootstrapper.add_cpue()
    assert "CPUE" in obtained_cpue.columns
    expected_cpue = [19.5, 19, 18.5, 18, 17.5, 17]
    assert (obtained_cpue.CPUE == expected_cpue).all()
