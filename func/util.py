import pandas as pd

def data_transform(product, file_path):
    df_old = pd.read_csv(file_path, parse_dates=['DATE'])
    df = df_old.copy()
    targets = ['R_DEC', 'R_KO', 'R_SUB', 'B_DEC', 'B_KO', 'B_SUB', 'WINNER']

    df = df.convert_dtypes()
    df.rename(columns={'KO/TKO': 'KO'}, inplace=True)
    df['DECISION'] = ((df.DECISION_SPLIT + df.DECISION_MAJORITY + df.DECISION_UNANIMOUS) > 0).astype(int)

    df['R_DEC'] = df.WINNER.astype(bool) * df.DECISION.astype(bool)
    df['R_KO'] = df.WINNER.astype(bool) * df.KO.astype(bool)
    df['R_SUB'] = df.WINNER.astype(bool) * df.SUBMISSION.astype(bool)
    df['B_DEC'] = ~df.WINNER.astype(bool) * df.DECISION.astype(bool)
    df['B_KO'] = ~df.WINNER.astype(bool) * df.KO.astype(bool)
    df['B_SUB'] = ~df.WINNER.astype(bool) * df.SUBMISSION.astype(bool)
    df[targets] = df[targets].astype(int)

    rename_dict = {'WOMEN\'S BANTAMWEIGHT':'WOMENS BANTAMWEIGHT',
                   'WOMEN\'S FEATHERWEIGHT': 'WOMENS FEATHERWEIGHT',
                    'WOMEN\'S FLYWEIGHT': 'WOMENS FLYWEIGHT',
                    'WOMEN\'S STRAWWEIGHT': 'WOMENS STRAWEIGHT'}

    df.rename(columns=rename_dict, inplace=True)

    print(df.columns)

    print("SANITY CHECK")
    print("Name should be Kelvin Cattar, WINNER=1, R_DEC=1")
    print(df[['R_NAME', 'WINNER', 'R_DEC', 'R_KO', 'B_DEC']].iloc[0])
    df.to_csv(product, index = False)

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

def split_train_test_op_alt(upstream, product, test_ratio):
    dataframe = pd.read_csv(upstream['features-alt']['data'])
    train_set, test_set = split_train_test(dataframe, test_ratio)
    train_set.to_csv(product['train'], index = False)
    test_set.to_csv(product['test'], index = False)
