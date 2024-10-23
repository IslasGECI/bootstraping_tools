import pandas as pd
from bootstrapping_tools import Bootstrap_from_time_series_parametrizer


def get_df(file_path):
    with open(file_path) as file:
        df = pd.read_csv(file)
    return df


df = get_df("tests/data/dcco_laal_gumu_burrows_data.csv")
bootstrap_number = 10


def test_intervals_from_p_values_and_alpha():
    dcco = df[df["Nombre_en_ingles"] == "Double-crested Cormorant"]
    parametrizer = Bootstrap_from_time_series_parametrizer(blocks_length=2, N=bootstrap_number)
    parametrizer.set_data(dcco)
    assert parametrizer.parameters["alpha"] == 0.05
