
import os
import urllib.request

import pandas as pd


# Main source for the training data
DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
# Local file
DATA_FILE = 'data/OxCGRT_latest.csv'


def download_historical(url=DATA_URL, file=DATA_FILE):
    """Download data located at 'url' and save them at location 'file'

    >>> from transat.data.load import download_historical
    >>> download_historical()
    """
    if not os.path.exists('data'):
        os.mkdir('data')
    urllib.request.urlretrieve(url, file)

def load_historical(file=DATA_FILE):
    """Load historical data located at 'file'

    >>> from transat.data.load import load_historical
    >>> df = load_historical()
    """
    # Load historical data from local file
    df = pd.read_csv(file,
        parse_dates=['Date'],
        encoding="ISO-8859-1",
        dtype={"RegionName": str,
               "RegionCode": str},
        error_bad_lines=False)
    return df

def load_removed_GeoID():
    """Load dataframe of removed GeoID
    """
    m_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(m_path, "assets", "removed_geoid.csv")

    df = pd.read_csv(csv_path)

    return df

def load_population():
    """Load dataframe of populations
    """
    m_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(m_path, "assets", "geoid_population_regions.csv")

    df = pd.read_csv(csv_path)

    return df