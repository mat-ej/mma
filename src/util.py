import pandas as pd

def split_train_test(dataframe, test_ratio):
    test_size = int(dataframe.shape[0] * test_ratio)
    train_set = dataframe.iloc[test_size:]
    test_set = dataframe.iloc[0:test_size]
    return train_set, test_set

def split_train_test_op(upstream, product, test_ratio):
    dataframe = pd.read_csv(upstream['features']['data'])
    train_set, test_set = split_train_test(dataframe, test_ratio)
    train_set.to_csv(product['train'], index = False)
    test_set.to_csv(product['test'], index = False)