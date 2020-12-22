import os
import stat
import shutil
import urllib.request
from transatlantic.utils import load_dataset, get_actual_cases
from transatlantic.constants import WINDOW_SIZE, TRAINING_LAST_DATE
import pandas as pd


def onerror(func, path, exc_info):
    """
    Error handler for ``shutil.rmtree``.

    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.

    Usage : ``shutil.rmtree(path, onerror=onerror)``
    """
    import stat
    if not os.access(path, os.W_OK):
        # Is the error an access error ?
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise

def update_repo(repo='leaf-ai/covid-xprize',target_folder='covid-xprize-uptodate', overwrite=True):
    if os.path.exists(f'./{target_folder}'):
        if overwrite:
            shutil.rmtree(f'./{target_folder}', onerror=onerror)
            print(f'Overwriting...')
        else:
            print(f'{target_folder} already exists and not overwriting. Done.')
            return
    else:
        print(f'Cloning...')
    # Clone repo
    os.system(f'cmd /c "git clone https://github.com/{repo}.git {target_folder}"')
    # Delete .git dir and .gitignore to avoid errors in current repo
    shutil.rmtree(f'./{target_folder}/.git', onerror=onerror)
    os.remove(f'./{target_folder}/.gitignore')
    print(f'Successfully cloned to {target_folder}')
    
def download_csv(CSV_URL, fname, dest_folder='./data'):
    # Local file: base dir, data
    os.makedirs(dest_folder, exist_ok=True)
    DATA_FILE = f'{dest_folder}/{fname}.csv'
    urllib.request.urlretrieve(CSV_URL, DATA_FILE)
    print(f'{fname} updated to {dest_folder}')
    
def load_dataset_filtered(data_file='./data/OxCGRT_latest.csv', 
               countries_regions_file='./covid-xprize-uptodate/countries_regions.csv'):

    df = load_dataset(data_file)
    
    df_countries = pd.read_csv(countries_regions_file, 
                               sep=",",
                               dtype={"CountryName": str,
                                       "RegionName": str})
    df_countries['RegionName'] = df_countries['RegionName'].fillna('')
    
    # Kept rows from countries_regions_file 
    kept_countries = df_countries['CountryName'].unique()
    
    # Countries with regions have not empty RegionName
    countries_with_regions = df_countries[df_countries['RegionName'] != '']['CountryName'].unique()
    
    # Countries without regions are the difference between two previous sets
    countries_without_regions = list(set(kept_countries) - set(countries_with_regions))
    
    # Kept regions include '' for whole country in countries_with_regions 
    kept_regions = df_countries['RegionName'].unique()
    
    # Filter: either dont'have regions --> RegionName is Nan
    #                have regions --> RegionName in kept_regions
    df = df[(df['CountryName'].isin(countries_without_regions) & (df['RegionName'] == '')) ^
            ((df['CountryName'].isin(countries_with_regions)) & (df['RegionName'].isin(kept_regions)))]    
    
    df.reset_index(drop=True, inplace=True)
    
    return df

def add_deaths(df, owid_csv_path='./data/OWID_latest.csv'):
    df_deaths = pd.read_csv(owid_csv_path, 
                            sep=",",
                            parse_dates=['date'],
                            encoding="ISO-8859-1",
                            dtype={'location': str},
                            error_bad_lines=False)
    
    # Keep only cumulated deaths 
    df_deaths = df_deaths[['location', 'date', 'total_deaths']]
    
    # Rename for cosistent column names
    df_deaths.rename(columns={'location': 'CountryName', 'date': 'Date', 'total_deaths': 'ConfirmedDeaths'}, inplace=True)
    
    # Merge death column
    df = df.merge(df_deaths, how='left')
    df.reset_index(drop=True, inplace=True)
    
    # Oldest confirmed deaths is 0, ffill from that
    min_date = df.Date.min()
    df.ConfirmedDeaths.where(df.Date != min_date, 0.0, inplace=True)
    df.ConfirmedDeaths.ffill(inplace=True)
    
    # Diff deaths and compute moving average
    df["ActualDailyNewDeaths"] = df.groupby("GeoID")["ConfirmedDeaths"].diff().fillna(0)
    df["ActualDailyNewDeaths7DMA"] = df.groupby(
        "GeoID")['ActualDailyNewDeaths'].rolling(
        WINDOW_SIZE, center=False).mean().reset_index(0, drop=True)
    
    # Oldest rolling new deaths is 0, ffill from that
    df.ActualDailyNewDeaths7DMA.where(df.Date != min_date, 0.0, inplace=True)
    df.ActualDailyNewDeaths7DMA.ffill(inplace=True)    
    
    return df

def get_features(country_list, owid_csv_path='./data/OWID_latest.csv'):
    
    # Open OWID dataset
    df_owid = pd.read_csv(owid_csv_path, 
                          sep=",",
                          parse_dates=['date'],
                          encoding="ISO-8859-1",
                          dtype={'location': str},
                          error_bad_lines=False)
    
    # Filter by countries in country_list
    df_owid = df_owid[df_owid['location'].isin(country_list)]
    df_owid.drop(['iso_code', 'continent'], axis=1, inplace=True)
    
    # Group by country
    grouped = df_owid.groupby('location')
    
    # Discard columns with more than 2 distinct values (actual value & nan)
    attribute_columns = [c for c in grouped.nunique().columns if max(grouped.nunique()[c]) < 2]
    
    # Keep location to identify country
    attribute_columns.append('location')
    
    # Get max value per column, drop non-unique columns and rename to CountryName
    attr_df = grouped.max(numeric_only=True).reset_index()
    attr_df.drop([c for c in attr_df.columns if c not in attribute_columns], axis=1, inplace=True)
    attr_df.rename(columns={'location': 'CountryName'}, inplace=True)

    return attr_df