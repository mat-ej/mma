import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import numpy as np


# def var_select(data, model):
#     weights = get_sample_weights(X_train, daily_discount_rate)
#     var_selector_model = var_selector_model.fit(X_train, y_train)
#     var_selector = SelectFromModel(var_selector_model, prefit=True)
#     X_train_new = var_selector.transform(X_train)
#
#     # plot_fea_importance(var_selector_model, X_train)
#
#     selected_vars_ = X_train.columns[var_selector.get_support()].to_list()


def get_sample_weights(dates: pd.Series, daily_discount_rate: float) -> np.array:
    day_deltas = (dates.iloc[0] - dates).astype('timedelta64[D]').astype(float)
    weights = np.power(np.full(shape=(len(dates),), fill_value=(1 - daily_discount_rate)), day_deltas)
    return weights

def logit_analysis(target_column: str, feature_columns:list , features: pd.DataFrame):
    df_ols = features[[target_column] + feature_columns].dropna()
    feature_sc = StandardScaler()
    target_sc = StandardScaler()

    # scaled_target = target_sc.fit_transform(df_ols[[target_column]])
    unscaled_target = df_ols[target_column]
    scaled_features = feature_sc.fit_transform(df_ols[feature_columns])

    features_df = pd.DataFrame(scaled_features, index=df_ols.index, columns=df_ols[feature_columns].columns)
    target_df = pd.DataFrame(unscaled_target, index=df_ols.index)

    mod = sm.Logit(target_df, features_df)
    fii = mod.fit(maxiter=1000)
    print(fii.summary())

def logit_analysis_reg(target_column: str, feature_columns:list , features: pd.DataFrame):
    df_ols = features[[target_column] + feature_columns].dropna()
    feature_sc = StandardScaler()
    target_sc = StandardScaler()

    # scaled_target = target_sc.fit_transform(df_ols[[target_column]])
    unscaled_target = df_ols[target_column]
    scaled_features = feature_sc.fit_transform(df_ols[feature_columns])

    features_df = pd.DataFrame(scaled_features, index=df_ols.index, columns=df_ols[feature_columns].columns)
    target_df = pd.DataFrame(unscaled_target, index=df_ols.index)

    mod = sm.Logit(target_df, features_df)
    fii = mod.fit_regularized(method='l1', maxiter=1000)
    print(fii.summary())
