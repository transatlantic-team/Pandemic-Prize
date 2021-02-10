import os
import base64
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd

# Set to True to run locally
TEST_MODE = False

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

NUM_PREV_DAYS_TO_INCLUDE = 6
WINDOW_SIZE = 7


def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

def get_actual_cases(df, start_date, end_date):
    # 1 day earlier to compute the daily diff
    start_date_for_diff = start_date - pd.offsets.Day(WINDOW_SIZE)
    actual_df = df[["CountryName", "RegionName", "Date", "ConfirmedCases"]]
    # Filter out the data set to include all the data needed to compute the diff
    actual_df = actual_df[(actual_df.Date >= start_date_for_diff) & (actual_df.Date <= end_date)]
    # Add GeoID column that combines CountryName and RegionName for easier manipulation of data
    # np.where usage: if A then B else C
    actual_df["GeoID"] = np.where(actual_df["RegionName"].isnull(),
                                  actual_df["CountryName"],
                                  actual_df["CountryName"] + ' / ' + actual_df["RegionName"])
    actual_df.sort_values(by=["GeoID","Date"], inplace=True)
    # Compute the diff
    actual_df["ActualDailyNewCases"] = actual_df.groupby("GeoID")["ConfirmedCases"].diff().fillna(0)
    # Compute the 7 day moving average
    actual_df["ActualDailyNewCases7DMA"] = actual_df.groupby(
        "GeoID")['ActualDailyNewCases'].rolling(
        WINDOW_SIZE, center=False).mean().reset_index(0, drop=True)
    return actual_df

def get_data(data_file):
    latest_df = pd.read_csv(data_file,
                            parse_dates=['Date'],
                            encoding="ISO-8859-1",
                            dtype={"RegionName": str,
                                   "RegionCode": str},
                            error_bad_lines=False)
    latest_df["RegionName"] = latest_df["RegionName"].fillna("")
    # Fill any missing NPIs by assuming they are the same as previous day, or 0 if none is available
    latest_df.update(latest_df.groupby(['CountryName', 'RegionName'])[NPI_COLUMNS].ffill().fillna(0))
    return latest_df

def get_predictions_from_file(predictor_name, predictions_file, ma_df):
    preds_df = pd.read_csv(predictions_file,
                           parse_dates=['Date'],
                           encoding="ISO-8859-1",
                           error_bad_lines=False)
    preds_df["RegionName"] = preds_df["RegionName"].fillna("")
    preds_df["PredictorName"] = predictor_name
    preds_df["Prediction"] = True

    # Append the true number of cases before start date
    ma_df["PredictorName"] = predictor_name
    ma_df["Prediction"] = False
    preds_df = ma_df.append(preds_df, ignore_index=True)

    # Add GeoID column that combines CountryName and RegionName for easier manipulation of data
    # np.where usage: if A then B else C
    preds_df["GeoID"] = np.where(preds_df["RegionName"].isnull(),
                                 preds_df["CountryName"],
                                 preds_df["CountryName"] + ' / ' + preds_df["RegionName"])
    # Sort
    preds_df.sort_values(by=["GeoID","Date"], inplace=True)
    # Compute the 7 days moving average for PredictedDailyNewCases
    preds_df["PredictedDailyNewCases7DMA"] = preds_df.groupby(
        "GeoID")['PredictedDailyNewCases'].rolling(
        WINDOW_SIZE, center=False).mean().reset_index(0, drop=True)

    # Put PredictorName first
    preds_df = preds_df[["PredictorName"] + [col for col in preds_df.columns if col != "PredictorName"]]
    return preds_df


if __name__ == '__main__':

    # Get arguments
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Define the dates
    start_date_str = "2020-08-01"
    end_date_str = "2020-08-30"
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

    # Define submission path and data path
    if TEST_MODE:
        submission_path = os.path.abspath(input_dir)
        data_path = os.path.abspath('../dataset')
        reference_path =  os.path.abspath('../reference')
    else:
        submission_path = os.path.abspath(os.path.join(input_dir, 'res'))
        data_path = os.path.abspath(os.path.join(input_dir, 'ref'))
        reference_path = data_path

    # Store predictions
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Train the model
    if not os.path.exists(os.path.join(submission_path, "train.py")):
        print("Skyping model training")
    else:
        print("Training the model")
        subprocess.call(["python",
                         os.path.join(submission_path, "train.py"),
                         "-data", os.path.join(data_path, 'training.csv')], cwd=submission_path)
        print("Storing the model on output path")
        zipf = zipfile.ZipFile(os.path.join(output_dir,'models.zip'), 'w', zipfile.ZIP_BZIP2)
        zipdir(os.path.join(submission_path, 'models'), zipf)
        zipf.close()
        with open(os.path.join(output_dir,'detailed_results.html'), 'w') as fd_res:
            fd_res.write('<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">\n')
            fd_res.write('<HTML>\n')
            fd_res.write('<HEAD>\n')
            fd_res.write('<TITLE>xPrize Detailed Results</TITLE>\n')
            fd_res.write('</HEAD>\n')
            fd_res.write('<BODY>\n')
            fd_res.write('<a download="models.zip" href="data:application/zip;base64,{}">Download models</a>\n'.format(
                base64.b64encode(open(os.path.join(output_dir,'models.zip'), 'rb').read()).decode())
            )
            fd_res.write('</BODY>\n')
            fd_res.write('</HTML>')

    # Make predictions
    print("Start Predicting 30 days")
    pred30_file = os.path.abspath(os.path.join(output_dir, 'predictions30.csv'))
    subprocess.call(["python",
                     os.path.join(submission_path, "predict.py"),
                     "-s", start_date_str,
                     "-e", end_date_str,
                     "-ip", os.path.join(data_path, "historical_ip.csv"),
                     "-o", pred30_file], cwd=submission_path)

    # Get ground truth data
    latest_df = get_data(os.path.join(reference_path, 'OxCGRT_latest.csv'))
    actual_df = get_actual_cases(latest_df, start_date, end_date)
    ma_df = actual_df[actual_df["Date"] < start_date]
    ma_df = ma_df[["CountryName", "RegionName", "Date", "ActualDailyNewCases"]]
    ma_df = ma_df.rename(columns={"ActualDailyNewCases": "PredictedDailyNewCases"})

    # Get predictions
    preds_df = get_predictions_from_file("Submission", pred30_file, ma_df.copy())

    # Evaluate predictions
    ranking_df = pd.DataFrame()
    merged_df = actual_df.merge(preds_df, on=['CountryName', 'RegionName', 'Date', 'GeoID'], how='left')
    ranking_df = ranking_df.append(merged_df)
    ranking_df['DiffDaily'] = (ranking_df["ActualDailyNewCases"] - ranking_df["PredictedDailyNewCases"]).abs()
    ranking_df['Diff7DMA'] = (ranking_df["ActualDailyNewCases7DMA"] - ranking_df["PredictedDailyNewCases7DMA"]).abs()

    # Compute the cumulative sum of 7DMA errors
    ranking_df['CumulDiffDaily'] = ranking_df.groupby(["GeoID", "PredictorName"])['DiffDaily'].cumsum()
    ranking_df['CumulDiff7DMA'] = ranking_df.groupby(["GeoID", "PredictorName"])['Diff7DMA'].cumsum()

    # Keep only predictions (either Prediction == True) or on or after start_date
    ranking_df = ranking_df[ranking_df["Date"] >= start_date]

    # Take only final values at certain time steps
    ranking_df4 = ranking_df[ranking_df["Date"] < start_date + pd.DateOffset(4)]
    ranking_df7 = ranking_df[ranking_df["Date"] < start_date + pd.DateOffset(7)]
    ranking_df30 = ranking_df[ranking_df["Date"] < start_date + pd.DateOffset(30)]

    # Evaluate predictions
    mae4 = ranking_df4['CumulDiffDaily'].sum()/4.0
    mae7 = ranking_df7['CumulDiffDaily'].sum()/7.0
    mae30 = ranking_df30['CumulDiffDaily'].sum()/30.0

    output_filename = os.path.join(output_dir, 'scores.txt')
    output_file = open(output_filename, 'wb')
    output_file.write("Mae4: {}\n".format(mae4).encode())
    output_file.write("Mae7: {}\n".format(mae7).encode())
    output_file.write("Mae30: {}\n".format(mae30).encode())
    output_file.close()
