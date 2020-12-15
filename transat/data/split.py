def split_historical(df, split_date):
    """Split the historical dataframe in a training and testing datasets according to a 'split_date'.

    >>> from transat.data import HYPOTHETICAL_SUBMISSION_DATE
    >>> from transat.data.split import split_historical
    >>> df_train, df_test = split_historical(df, HYPOTHETICAL_SUBMISSION_DATE)

    Args:
        df (pd.DataFrame): the dataframe to split.
        split_date (np.datetime64): the split date.

    Returns:
        (df_train, df_test): where ``df_train`` contains all data <= ``split_date`` and ``df_test`` contains all data > ``split_date``.
    """
    df_train = df[df.Date <= split_date]
    df_test = df[df.Date > split_date]
    return df_train, df_test