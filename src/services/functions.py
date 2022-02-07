def get_odds(dataframe):
    return dataframe[['R_ODDS', 'B_ODDS']].values


def get_results(dataframe):
    return dataframe['WINNER'].values.reshape(-1,)


def split_train_test(dataframe, test_ratio):
    test_size = int(dataframe.shape[0] * test_ratio)
    train_set = dataframe.iloc[test_size:]
    test_set = dataframe.iloc[0:test_size]
    return train_set, test_set
