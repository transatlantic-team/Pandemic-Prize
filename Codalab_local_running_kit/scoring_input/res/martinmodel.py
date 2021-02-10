import os
import pickle
import numpy as np
import pandas as pd

# os.system("pip install statsmodels")


from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

def predict(start_date, end_date, ip_file, output_file):
        """
            Predict new cases between two dates

            :param start_date: First date of the interval in the format (yyyy-mm-dd)
            :type start_date: str
            :param end_date: Last date of the interval in the format (yyyy-mm-dd)
            :type end_date: str
            :param ip_file: Path to the Intervention Plan file
            :type ip_file: str
            :param output_file: File to write the output predictions
            :type output_file: str
            :param verbose: Whether to show traces for debug (True) or run in quiet mode (False)
            :type verbose: bool
        """
        start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date, format='%Y-%m-%d')

        steps = (end_date - start_date).days

        NPI_COLUMNS = ['C1_School closing',
               'C2_Workplace closing',
               'C3_Cancel public events',
               'C4_Restrictions on gatherings',
               'C5_Close public transport',
               'C6_Stay at home requirements',
               'C7_Restrictions on internal movement',
               'C8_International travel controls',
               'H1_Public information campaigns',
               'H2_Testing policy',
               'H3_Contact tracing',
               'H6_Facial Coverings']

        ID_COLUMNS = ['CountryName',
              'RegionName',
              'GeoID',
              'Date']

        # Load historical intervention plans, since inception
        hist_ips_df = pd.read_csv(ip_file,
                                  parse_dates=['Date'],
                                  encoding="ISO-8859-1",
                                  dtype={"RegionName": str},
                                  error_bad_lines=True)

        hist_ips_df['RegionName'] = hist_ips_df['RegionName'].fillna('')

        # Add GeoID column that combines CountryName and RegionName for easier manipulation of data",
        hist_ips_df['GeoID'] = hist_ips_df['CountryName'] + '__' + hist_ips_df['RegionName'].astype(str)

        # Fill any missing NPIs by assuming they are the same as previous day
        for npi_col in NPI_COLUMNS:
            hist_ips_df.update(hist_ips_df.groupby(['CountryName', 'RegionName'])[npi_col].ffill().fillna(0))

        # Intervention plans to forecast for: those between start_date and end_date
        ips_df = hist_ips_df[(hist_ips_df.Date >= start_date) & (hist_ips_df.Date < end_date)]

        geo_pred_dfs = []
        for g in ips_df.GeoID.unique():
            
            ips_gdf = ips_df[ips_df.GeoID == g]

            try:
                savedmodel = SARIMAXResults.load(f'./models_ARIMA/{g}.pkl')
                preds = savedmodel.forecast(steps=steps, exog=ips_gdf[NPI_COLUMNS])
                preds_newcases = np.ediff1d(preds, to_begin=0.0)

                # print(preds_newcases)
                preds_newcases = np.clip(preds_newcases, a_min=0.0, a_max=None)
            except FileNotFoundError:
                preds_newcases = np.zeros(steps)
            

            
            geo_pred_df = ips_gdf[ID_COLUMNS].copy()
            geo_pred_df['PredictedDailyNewCases'] = preds_newcases
            geo_pred_dfs.append(geo_pred_df)


        # Combine all predictions into a single dataframe
        pred_df = pd.concat(geo_pred_dfs)

        # Drop GeoID column to match expected output format
        pred_df = pred_df.drop(columns=['GeoID'])

        # Create the output path
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # Save to a csv file
        pred_df.to_csv(output_file, index=False)