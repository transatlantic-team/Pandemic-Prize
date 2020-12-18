import numpy as np

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
#     "StringencyIndex",
#     "NewCasesSmoothed14Days"
]


def preprocess_historical_basic(df, cases_col=CASES_COL, npi_cols=NPI_COLS):
    """Create a copy of preprocessed data."""

    df = df.copy()

    # Add RegionID column that combines CountryName and RegionName for easier manipulation of data
    df["GeoID"] = df["CountryName"] + "__" + df["RegionName"].astype(str)

    # Add new cases column
    df["NewCases"] = df.groupby("GeoID").ConfirmedCases.diff().fillna(0)
    df.loc[df.NewCases < 0, "NewCases"] = 0

    df["NewCasesSmoothed7Days"] = df.groupby("GeoID").NewCases.rolling(7, win_type="boxcar").mean().fillna(0).reset_index(level=0, drop=True) #.reset_index()
    
#     df["NewCasesSmoothed14Days"] = df.groupby("GeoID").NewCases.rolling(14, win_type="boxcar").mean().fillna(0).reset_index(level=0, drop=True)
    
#     norm_cols = ["C1_School closing", "C2_Workplace closing", "C3_Cancel public events", "C4_Restrictions on gatherings",
#         "C5_Close public transport", "C6_Stay at home requirements", "C7_Restrictions on internal movement", 
#         "C8_International travel controls"]
#     df[norm_cols] /= 4
    

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


def dataframe_to_array(df, nb_lookback_days=30, nb_lookahead_days=30, sequence_format=False, cases_col=CASES_COL, npi_cols=NPI_COLS, neg_npis=True):
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
    X_samples = []
    y_samples = []
    geo_ids = df.GeoID.unique()
    for g in geo_ids:
        gdf = df[df.GeoID == g]
        all_case_data = np.array(gdf[cases_col])
        all_npi_data = np.array(gdf[npi_cols])

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
#             y_sample = all_case_data[d + 1]
            y_sample = all_case_data[d+1:d+1+nb_lookahead_days]
            X_samples.append(X_sample)
            y_samples.append(y_sample)

    X_samples = np.array(X_samples)

    y_samples = np.array(y_samples)
    if sequence_format:
#         y_samples = y_samples.reshape(-1, nb_lookahead_days, 1)
        pass
    else:
        y_samples = y_samples.flatten()

    return (X_samples, y_samples), (X_cols, y_col)