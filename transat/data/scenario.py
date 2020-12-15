import numpy as np

from .transform import CASES_COL, NPI_COLS


def generate_scenario(
    df_train,
    df_test,
    geo_id,
    nb_lookback_days=30,
    nb_future_days=30,
    sequence_format=False,
    neg_npis=True,
    cases_col=CASES_COL,
    npi_cols=NPI_COLS,
):
    # Create training data across one country for predicting one day ahead

    gdf = df_train[df_train.GeoID == geo_id]
    all_case_data = np.array(gdf[cases_col])
    all_npi_data = np.array(gdf[npi_cols])

    # Create one sample for each day where we have enough data
    # Each sample consists of cases and npis for previous nb_lookback_days
    X_cases = all_case_data[-nb_lookback_days:]

    # Take negative of npis to support positive weight constraint in Lasso.
    X_npis = all_npi_data[-nb_lookback_days:]
    if neg_npis:
        X_npis = -X_npis

    if sequence_format:
        X_sample = np.concatenate([X_cases, X_npis], axis=1)
    else:
        # Flatten all input data so it fits Lasso input format.
        X_sample = np.concatenate([X_cases.flatten(), X_npis.flatten()])


    X_input = np.array([X_sample])

    gdf = df_test[df_test.GeoID == geo_id]
    all_case_data = np.array(gdf[cases_col])

    y_output = all_case_data[:nb_future_days]
    if sequence_format:
        y_output = y_output.reshape(1,-1,1)
    else:
        y_output = y_output.flatten()

    return X_input, y_output