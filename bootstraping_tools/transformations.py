import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from tqdm import tqdm


def power_law(T, Lambda, No):
    return No * np.power(Lambda, T)


def lambda_calculator(
    temporadas, maximo_nidos, max_iter=2000, lower_bounds=0, lambda_upper_bound=50
):
    temporadas = np.array(temporadas)
    numero_agno = temporadas - temporadas[0]
    maximo_nidos = np.array(maximo_nidos)
    popt, pcov = curve_fit(
        power_law,
        numero_agno,
        maximo_nidos,
        maxfev=max_iter,
        bounds=((lower_bounds, lower_bounds), (lambda_upper_bound, np.inf)),
    )
    return popt


def remove_distribution_outliers(data):
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    mask = abs(data - mean) < std * 5
    return data[mask]


def seasons_from_date(data):
    seasons = data["Fecha"].str.split("/", n=2, expand=True)
    return np.array(seasons[2])


def boostrapping_feature(data, N=2000):
    dataframe = pd.DataFrame(data)
    bootstrap_data = []
    for i in range(N):
        resampled_data = dataframe.sample(n=1, random_state=i)
        bootstrap_data.append(resampled_data.iloc[0][0])
    return bootstrap_data


def lambdas_from_bootstrap_table(dataframe):
    lambdas_bootstraps = []
    seasons = np.array(dataframe.columns.values, dtype=int)
    N = len(dataframe)
    print("Calculating bootstrap growth rates distribution:")
    for i in tqdm(range(N)):
        fitting_result = lambda_calculator(seasons, dataframe.T[i].values)
        lambdas_bootstraps.append(fitting_result[0])
    return lambdas_bootstraps


def lambdas_bootstrap_from_dataframe(dataframe, column_name, N=20, return_distribution=False):
    bootstraped_data = pd.DataFrame()
    lambdas_bootstraps = []
    seasons = dataframe.sort_values(by="Temporada").Temporada.unique()
    print("Calculating samples per season:")
    for season in tqdm(seasons):
        data_per_season = dataframe[dataframe.Temporada == season]
        bootstraped_data[season] = boostrapping_feature(data_per_season[column_name], N)
    lambdas_bootstraps = lambdas_from_bootstrap_table(bootstraped_data)
    if return_distribution:
        return lambdas_bootstraps, np.percentile(lambdas_bootstraps, [2.5, 50, 97.5])
    else:
        return np.percentile(lambdas_bootstraps, [2.5, 50, 97.5])


def get_bootstrap_interval(bootrap_interval):
    inferior_limit = np.around(bootrap_interval[1] - bootrap_interval[0], decimals=2)
    superior_limit = np.around(bootrap_interval[2] - bootrap_interval[1], decimals=2)
    bootrap_interval = np.around(bootrap_interval, decimals=2)
    return [inferior_limit, bootrap_interval[1], superior_limit]


def bootstrap_from_time_series(dataframe, column_name, N=20, return_distribution=False):
    lambdas_bootstraps = []
    print("Calculating bootstrap growth rates distribution:")
    for i in tqdm(range(N)):
        resampled_data = dataframe.sample(
            n=len(dataframe), replace=True, random_state=i
        ).sort_index()
        fitting_result = lambda_calculator(resampled_data["Temporada"], resampled_data[column_name])
        lambdas_bootstraps.append(fitting_result[0])
    lambdas_bootstraps = remove_distribution_outliers(lambdas_bootstraps)
    if return_distribution:
        return lambdas_bootstraps, np.percentile(lambdas_bootstraps, [2.5, 50, 97.5])
    else:
        return np.percentile(lambdas_bootstraps, [2.5, 50, 97.5])


def calculate_p_values(distribution):
    distribution = np.array(distribution)
    mask = distribution < 1
    mask2 = distribution > 1
    return mask.sum() / len(distribution), mask2.sum() / len(distribution)

