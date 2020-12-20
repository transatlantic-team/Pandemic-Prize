import os
import numpy as np
import pandas as pd

from transat.data.load import load_removed_GeoID, load_population

# CASES_COL = ["NewCases"]
CASES_COL = ["NewCasesSmoothed7Days"]
NPI_COLS = [
    "C1_School closing",
    "C2_Workplace closing",
    "C3_Cancel public events",
    "C4_Restrictions on gatherings",
    "C5_Close public transport",
    "C6_Stay at home requirements",
    "C7_Restrictions on internal movement",
    "C8_International travel controls",
    "H1_Public information campaigns",
    "H2_Testing policy",
    "H3_Contact tracing",
    "H6_Facial Coverings",
]
CONTEXT_COLS = [
    'CountryName',
    'RegionName',
    'GeoID',
    'Date',
    'ConfirmedCases',
    'ConfirmedDeaths',
    'Population'
]
WINDOW_SIZE = 7

def remove_geoid(df):
    """Remove some GeoID not considered for the competition"""
    df_rem_GeoID = load_removed_GeoID()
    return  df[~df.GeoID.isin(df_rem_GeoID.GeoID)]

def normalize_by_population(df, columns):
    df_pop = load_population()

    geoid_to_pop = {g:p for g,p in zip(df_pop.GeoID, df_pop.Population)}

    for geoid in df.GeoID.unique():
        df.loc[df.GeoID == geoid, columns] /= geoid_to_pop[geoid]


def preprocess_historical_basic(df, cases_col=CASES_COL, npi_cols=NPI_COLS, norm_npis=True, norm_by_pop=True):
    """Create a copy of preprocessed data."""

    df = df.copy()

    # Add RegionID column that combines CountryName and RegionName for easier manipulation of data
    df["GeoID"] = df["CountryName"] + "__" + df["RegionName"].astype(str)

    # Remove GeoID not considered in the competition
    df = remove_geoid(df)

    # Add new cases column
    df["NewCases"] = df.groupby("GeoID").ConfirmedCases.diff().fillna(0)
    df.loc[df.NewCases < 0, "NewCases"] = 0

    df["NewCasesSmoothed7Days"] = df.groupby("GeoID").NewCases.rolling(7, win_type=None).mean().fillna(0).reset_index(level=0, drop=True) #.reset_index()

#     df["NewCasesSmoothed14Days"] = df.groupby("GeoID").NewCases.rolling(14, win_type="boxcar").mean().fillna(0).reset_index(level=0, drop=True)

    if norm_npis:
        df[NPI_COLS] /= 4

    if norm_by_pop:
        normalize_by_population(df, cases_col)

    # Keep only columns of interest
    id_cols = ["CountryName", "RegionName", "GeoID", "Date", "NewCases"]

    df = df[id_cols + cases_col + npi_cols]

    # Fill any missing case values by interpolation and setting NaNs to 0
    df.update(
        df.groupby("GeoID").NewCases.apply(lambda group: group.interpolate()).fillna(0)
    )

    # Fill any missing NPIs by assuming they are the same as previous day
    for npi_col in npi_cols:
        df.update(df.groupby("GeoID")[npi_col].ffill().fillna(0))

    return df

def fill_missing_values(df, npi_cols=NPI_COLS):
    """
    # Fill missing values by interpolation, ffill, and filling NaNs
    :param df: Dataframe to be filled
    """
    df.update(df.groupby('GeoID').ConfirmedCases.apply(
        lambda group: group.interpolate(limit_area='inside')))
    
    # Drop country / regions for which no number of cases is available
    df.dropna(subset=['ConfirmedCases'], inplace=True)
    df.update(df.groupby('GeoID').ConfirmedDeaths.apply(
        lambda group: group.interpolate(limit_area='inside')))
    
    # Drop country / regions for which no number of deaths is available
    df.dropna(subset=['ConfirmedDeaths'], inplace=True)
    for npi_column in npi_cols:
        df.update(df.groupby('GeoID')[npi_column].ffill().fillna(0))

def preprocess_historical_lstm(df, context_cols=CONTEXT_COLS, npi_cols=NPI_COLS, window_size=WINDOW_SIZE):
    df = df.copy()

    # Add RegionID column that combines CountryName and RegionName for easier manipulation of data
    df["GeoID"] = df["CountryName"] + "__" + df["RegionName"].astype(str)

    # Remove GeoID not considered in the competition
    df = remove_geoid(df)

    # Additional context df (e.g Population for each country)
    df_pop = load_population()

    geoid_to_pop = {g:p for g,p in zip(df_pop.GeoID, df_pop.Population)}

    for geoid in df.GeoID.unique():
        df.loc[df.GeoID == geoid, "Population"] = geoid_to_pop[geoid]

    # Drop countries with no population data
    df.dropna(subset=['Population'], inplace=True)

    #  Keep only needed columns
    columns = context_cols + npi_cols
    df = df[columns]

    # Fill in missing values
    fill_missing_values(df, npi_cols)

    # Compute number of new cases and deaths each day
    df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)
    df['NewDeaths'] = df.groupby('GeoID').ConfirmedDeaths.diff().fillna(0)

    # Replace negative values (which do not make sense for these columns) with 0
    df['NewCases'] = df['NewCases'].clip(lower=0)
    df['NewDeaths'] = df['NewDeaths'].clip(lower=0)

    # Compute smoothed versions of new cases and deaths each day
    df['SmoothNewCases'] = df.groupby('GeoID')['NewCases'].rolling(
        window_size, center=False).mean().fillna(0).reset_index(0, drop=True)
    df['SmoothNewDeaths'] = df.groupby('GeoID')['NewDeaths'].rolling(
        window_size, center=False).mean().fillna(0).reset_index(0, drop=True)

    # Compute percent change in new cases and deaths each day
    df['CaseRatio'] = df.groupby('GeoID').SmoothNewCases.pct_change(
    ).fillna(0).replace(np.inf, 0) + 1
    df['DeathRatio'] = df.groupby('GeoID').SmoothNewDeaths.pct_change(
    ).fillna(0).replace(np.inf, 0) + 1

    # Add column for proportion of population infected
    df['ProportionInfected'] = df['ConfirmedCases'] / df['Population']

    # Create column of value to predict
    df['PredictionRatio'] = df['CaseRatio'] / (1 - df['ProportionInfected'])

    return df

OTHER_COLS = [
    "ProportionInfected",
    "SmoothNewCases",
    "ConfirmedCases"
]

def dataframe_to_array(df, nb_lookback_days=30, nb_lookahead_days=30, sequence_format=False, cases_col=CASES_COL, npi_cols=NPI_COLS, neg_npis=True, other_cols=None):
    """Process dataframe to return an array formated for a training procedure.

    Args:
        df ([type]): [description]
        nb_lookback_days (int, optional): number of past days to use to make predictions. Defaults to 30.
        sequence_format (bool, optional): If ``True`` will return data with shape (nb_data_points, nb_lookback_days, feature_size) else will return data with shape (nb_data_points, nb_lookback_days*feature_size). Defaults to False.
        cases_col (list, optional): len(cases_col) must be 1. Contains the column name that we want to keep for the number of confirmed cases. Defaults to CASES_COL.
        npi_cols (list, optional): [description]. Defaults to NPI_COLS.

    Returns:
        (X_samples, y_samples), (X_cols, y_col): [description]
    """
    assert len(cases_col) == 1, "len(cases_col) must be 1!"

    # Create training data across all countries for predicting one day ahead
    X_cols = cases_col + npi_cols
    y_col = cases_col
    o_cols = other_cols
    X_samples = []
    y_samples = []
    X_o_samples = []
    y_o_samples = []
    geo_ids = df.GeoID.unique()
    for g in geo_ids:
        gdf = df[df.GeoID == g]
        all_case_data = np.array(gdf[cases_col])
        all_npi_data = np.array(gdf[npi_cols])
        
        if o_cols:
            all_o_data = np.array(gdf[o_cols])

        # Create one sample for each day where we have enough data
        # Each sample consists of cases and npis for previous nb_lookback_days
        nb_total_days = len(gdf)
        for d in range(nb_lookback_days, nb_total_days - 1 - nb_lookahead_days):
            X_cases = all_case_data[d - nb_lookback_days : d]
            X_npis = all_npi_data[d - nb_lookback_days : d]
            
            if neg_npis:
                # Take negative of npis to support positive
                # weight constraint in Lasso.
                X_npis = -X_npis

            if sequence_format:
                # Shape (nb_lookback_days, feature_size)
                X_sample = np.concatenate([X_cases, X_npis], axis=1)
            else:
                # Flatten all input data so it fits Lasso input format.
                # Shape (nb_lookback_days * feature_size)
                X_sample = np.concatenate([X_cases.flatten(), X_npis.flatten()])
                
            y_sample = all_case_data[d:d+nb_lookahead_days]
            X_samples.append(X_sample)
            y_samples.append(y_sample)
            if other_cols:
                X_o_sample  = all_o_data[d-nb_lookback_days : d]
                X_o_samples.append(X_o_sample)
                y_o_sample  = all_o_data[d:d+nb_lookahead_days]
                y_o_samples.append(y_o_sample)
            

    X_samples = np.array(X_samples)

    y_samples = np.array(y_samples)
    if not sequence_format:
        y_samples = y_samples.flatten()
    
    if o_cols:
        X_o_samples = np.array(X_o_samples)
        y_o_samples = np.array(y_o_samples)
        
        return (X_samples, y_samples, X_o_samples, y_o_samples), (X_cols, y_col, o_cols)
    else:
        return (X_samples, y_samples), (X_cols, y_col)



