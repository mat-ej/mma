import csv
import math

import numpy as np
import pandas as pd
import os
import datetime
import locale

from datetime import datetime, date

target_stats = [
    'WINNER',
    'WIN_BY',
    'NUM_OF_ROUNDS',
    'LAST_ROUND',
    'LAST_ROUND_TIME',
    'TOTAL_FIGHT_TIME',
    'WEIGHT_CATEGORY',
    'TITLE_BOUT',
    'REFEREE',
    'DATE',
    'LOCATION',
    'R_NAME',
    'R_AGE',
    'R_HEIGHT',
    'R_WEIGHT',
    'R_REACH',
    'R_KD',
    'R_SIG_STR',
    'R_SIG_STR_ATT',
    'R_TOTAL_STR',
    'R_TOTAL_STR_ATT',
    'R_TD',
    'R_TD_ATT',
    'R_SUB_ATT',
    'R_REV',
    'R_CTRL',
    'R_HEAD',
    'R_HEAD_ATT',
    'R_BODY',
    'R_BODY_ATT',
    'R_LEG',
    'R_LEG_ATT',
    'R_DISTANCE',
    'R_DISTANCE_ATT',
    'R_CLINCH',
    'R_CLINCH_ATT',
    'R_GROUND',
    'R_GROUND_ATT',
    'B_NAME',
    'B_AGE',
    'B_HEIGHT',
    'B_WEIGHT',
    'B_REACH',
    'B_KD',
    'B_SIG_STR',
    'B_SIG_STR_ATT',
    'B_TOTAL_STR',
    'B_TOTAL_STR_ATT',
    'B_TD',
    'B_TD_ATT',
    'B_SUB_ATT',
    'B_REV',
    'B_CTRL',
    'B_HEAD',
    'B_HEAD_ATT',
    'B_BODY',
    'B_BODY_ATT',
    'B_LEG',
    'B_LEG_ATT',
    'B_DISTANCE',
    'B_DISTANCE_ATT',
    'B_CLINCH',
    'B_CLINCH_ATT',
    'B_GROUND',
    'B_GROUND_ATT'

]


def read_files():
    current_file = os.path.abspath(os.path.dirname(__file__))
    filename_fights = os.path.join(current_file, '../../../data/new-raw/total_fight_data.csv')
    filename_fighters = os.path.join(current_file, '../../../data/new-raw/fighter_details.csv')
    filename_odds = os.path.join(current_file, '../../../data/new-raw/odds_data.csv')

    fights_df = pd.read_csv(filename_fights, sep=';')
    fighters_df = pd.read_csv(filename_fighters, sep=',')
    odds_df = pd.read_csv(filename_odds, sep=',')
    return fights_df, fighters_df, odds_df


def drop_draws_and_old_fights(fights_df):
    """
    Drop events prior to april 2001 (rule change), draws, weird formats, finds that ended in DQ
    :param fights_df:
    :return:
    """
    date_limit = date(2001, 4, 1)
    fights_df = fights_df[pd.to_datetime(fights_df['date'], format='%B %d, %Y') > pd.to_datetime(date_limit)]
    fights_df['Winner'].replace('', np.nan, inplace=True)
    fights_df.dropna(subset=['Winner'], inplace=True)
    fights_df = fights_df[fights_df['Format'] != '3 Rnd + OT (5-5-5-5)']
    fights_df = fights_df[fights_df['win_by'] != 'DQ']
    return fights_df


def convert_times_to_seconds(fights_df):
    columns = ["R_CTRL", "B_CTRL", 'last_round_time']
    # columns = ['last_round_time']

    def conv_to_sec(X):
        if ':' in X:
            return int(X.split(":")[0]) * 60 + int(X.split(":")[1])
        else:
            # if '--' means there was no time spent on the ground.
            # Taking a call here to consider this as 0 seconds
            return 0

    for column in columns:
        fights_df[column] = fights_df[column].apply(conv_to_sec)
    return fights_df


def time_to_total(fights_df):
    time_in_first_round = {
        "3 Rnd (5-5-5)": 5 * 60,
        "5 Rnd (5-5-5-5-5)": 5 * 60,
    }

    def get_total_time(row):
        if row["Format"] in time_in_first_round.keys():
            return (row["last_round"] - 1) * time_in_first_round[row["Format"]] + row["last_round_time"]

        else:
            print(row['Format'])

    fights_df['TOTAL_FIGHT_TIME'] = fights_df.apply(
        get_total_time, axis=1
    )
    return fights_df


def create_title_bout_feature(fights_df):
    fights_df["TITLE_BOUT"] = fights_df["Fight_type"].apply(lambda x: True if "Title Bout" in x else False)
    return fights_df


def create_weight_category(fights_df):
    def make_weight_class(X):
        weight_classes = [
            "Women's Strawweight",
            "Women's Bantamweight",
            "Women's Featherweight",
            "Women's Flyweight",
            "Lightweight",
            "Welterweight",
            "Middleweight",
            "Light Heavyweight",
            "Heavyweight",
            "Featherweight",
            "Bantamweight",
            "Flyweight",
            "Open Weight",
        ]

        for weight_class in weight_classes:
            if weight_class in X:
                return weight_class

        if X == "Catch Weight Bout" or "Catchweight Bout":
            return "Catch Weight"
        else:
            print('mistake: ', X)
            return "Open Weight"

    fights_df["WEIGHT_CATEGORY"] = fights_df["Fight_type"].apply(make_weight_class)
    return fights_df


def edit_win_by(fights_df):
    fights_df["win_by"] = fights_df["win_by"].apply(lambda x: 'KO/TKO' if ('KO' in x or 'TKO' in x) else x)
    return fights_df


def create_num_of_rounds(fights_df):
    fights_df["NUM_OF_ROUNDS"] = fights_df["Format"].apply(lambda x: int(x.split('Rnd')[0]))
    return fights_df


def edit_strikes(fights_df):
    mapping = {
        "R_SIG_STR.": 'R_SIG_STR',
        "B_SIG_STR.": 'B_SIG_STR',
        "R_TOTAL_STR.": 'R_TOTAL_STR',
        "B_TOTAL_STR.": 'B_TOTAL_STR',
    }

    columns = [
        "R_SIG_STR",
        "B_SIG_STR",
        "R_TOTAL_STR",
        "B_TOTAL_STR",
        "R_TD",
        "B_TD",
        "R_HEAD",
        "B_HEAD",
        "R_BODY",
        "B_BODY",
        "R_LEG",
        "B_LEG",
        "R_DISTANCE",
        "B_DISTANCE",
        "R_CLINCH",
        "B_CLINCH",
        "R_GROUND",
        "B_GROUND",
    ]
    fights_df.rename(columns=mapping, inplace=True)
    attempt_suffix = "_ATT"

    for column in columns:
        fights_df[column + attempt_suffix] = fights_df[column].apply(lambda X: int(X.split("of")[1]))
        fights_df[column] = fights_df[column].apply(lambda X: int(X.split("of")[0]))
    return fights_df


def drop_redundant_columns(fights_df):
    columns = [
        'R_SIG_STR_pct',
        'B_SIG_STR_pct',
        'R_TD_pct',
        'B_TD_pct',
        'Format',
        'Fight_type'
    ]
    fights_df.drop(columns=columns, inplace=True)
    return fights_df


def add_fighters_details(fights_df, fighters_df):
    def get_age(dob, date):
        if dob == '' or not (type(dob) == str):
            return 0
        birth_date = datetime.strptime(dob, '%b %d, %Y')
        fight_date = datetime.strptime(date, '%B %d, %Y')
        return math.floor((fight_date - birth_date).days / 365.25)

    def get_height(x):
        if x == '':
            return 0
        if type(x) == float:
            return round(x * 2.54, 1)
        feet = int(x.split('\'')[0])
        inch = int(x.split('\'')[1].split('\"')[0])
        inch += feet * 12
        return round(inch * 2.54, 1)

    def get_weight(x):
        if x == '':
            return 0
        if type(x) == float:
            return round(x * 0.45359, 1)
        return round(float(x.split(' ')[0]) * 0.45359, 1)

    def get_reach(x):
        if x == '':
            return 0
        if type(x) == float:
            return round(x * 2.54, 1)
        return round(int(x.split("\"")[0]) * 2.54, 1)

    def get_stuff(x):
        r_name = str(x['R_fighter']).strip()
        b_name = str(x['B_fighter']).strip()
        print(r_name)
        print(b_name)
        # print(str(x['R_fighter']))
        # print(fighters_df[fighters_df['fighter_name'].str.contains(str(x['R_fighter']))])
        r = fighters_df[fighters_df['fighter_name'].str.contains(r_name)].iloc[0]
        b = fighters_df[fighters_df['fighter_name'].str.contains(b_name)].iloc[0]

        # print(r)
        # print(b)
        r_age = get_age(r['DOB'], x['date'])
        b_age = get_age(b['DOB'], x['date'])
        r_height = get_height(r['Height'])
        b_height = get_height(b['Height'])
        r_reach = get_reach(r['Reach'])
        b_reach = get_reach(b['Reach'])

        # weight depends on time
        r_weight = get_weight(r['Weight'])
        b_weight = get_weight(b['Weight'])
        # ind = ['R_AGE', 'B_AGE', 'R_HEIGHT', 'B_HEIGHT', 'R_REACH', 'B_REACH']
        # [r_age, b_age, r_height, b_height, r_reach, b_reach]

        ret = pd.Series([r_age, r_height, r_weight, r_reach, b_age, b_height, b_weight, b_reach], index=['R_AGE', 'R_HEIGHT', 'R_WEIGHT', 'R_REACH', 'B_AGE', 'B_HEIGHT', 'B_WEIGHT', 'B_REACH'])
        # ret = pd.Series([r_age, b_age, r_height, b_height, r_reach, b_reach], index=ind)
        print(x['R_fighter'])
        print(x['B_fighter'])
        print(ret)
        return ret
    fights_df[['R_AGE', 'R_HEIGHT', 'R_WEIGHT', 'R_REACH', 'B_AGE', 'B_HEIGHT', 'B_WEIGHT', 'B_REACH']] = fights_df.apply(get_stuff, axis=1)
    return fights_df


def format_date(x):

    # laco = '1/15/2022'
    # datetime_object = datetime.strptime(laco, '%m/%d/%Y')
    # datetime_object = datetime.strptime(x, '%Y-%m-%d')
    if type(x) == float:
        return '2000-01-01'

    if '-' in x:
        datetime_object = datetime.strptime(x, '%Y-%m-%d')
    elif '/' in x:
        datetime_object = datetime.strptime(x, '%m/%d/%Y')
    else:
        print("BAD DATE FORMAT")
        return None

    str = datetime_object.strftime('%Y-%m-%d')

    # if len(spl[0]) < 2:
    #     spl[0] = '0' + spl[0]
    # if len(spl[1]) < 2:
    #     spl[1] = '0' + spl[1]
    return str


def reformat_odds_date(odds_df):
    odds_df['date'] = odds_df['date'].apply(format_date)
    return odds_df


def add_odds(fights_df, odds_df):
    def odds_to_decimal(x):
        if math.isnan(x):
            return float('nan')
        x = int(x)
        if x > 0:
            return round(1 + (x/100), 2)
        else:
            return round(1 - (100/x), 2)

    def get_odds(x):
        fight_date = pd.to_datetime(x['date'], format='%B %d, %Y')
        fight_night = pd.to_datetime(odds_df['date'], format='%Y-%m-%d') == fight_date
        if sum(fight_night) == 0:
            return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                             index=['R_ODDS', 'B_ODDS', 'R_DEC_ODDS', 'B_DEC_ODDS', 'R_SUB_ODDS', 'B_SUB_ODDS', 'R_KO_ODDS', 'B_KO_ODDS'])
        for index, fight in odds_df[fight_night].iterrows():
            if x['R_fighter'] == fight['R_fighter'] and x['B_fighter'] == fight['B_fighter']:
                r_odds = odds_to_decimal(fight['R_odds'])
                b_odds = odds_to_decimal(fight['B_odds'])

                r_dec_odds = odds_to_decimal(fight['r_dec_odds'])
                b_dec_odds = odds_to_decimal(fight['b_dec_odds'])
                r_sub_odds = odds_to_decimal(fight['r_sub_odds'])
                b_sub_odds = odds_to_decimal(fight['b_dec_odds'])
                r_ko_odds = odds_to_decimal(fight['r_ko_odds'])
                b_ko_odds = odds_to_decimal(fight['b_ko_odds'])

                s = pd.Series([r_odds, b_odds, r_dec_odds, b_dec_odds, r_sub_odds, b_sub_odds, r_ko_odds, b_ko_odds],
                           index=['R_ODDS', 'B_ODDS', 'R_DEC_ODDS', 'B_DEC_ODDS', 'R_SUB_ODDS', 'B_SUB_ODDS', 'R_KO_ODDS', 'B_KO_ODDS'])

                print(s)
                return s
                # return pd.Series([r_odds, b_odds], index=['R_ODDS', 'B_ODDS'])
    # for ind, x in fights_df.iterrows():
    #     print(row)

    test = fights_df.copy()
    test[['R_ODDS', 'B_ODDS', 'R_DEC_ODDS', 'B_DEC_ODDS', 'R_SUB_ODDS', 'B_SUB_ODDS', 'R_KO_ODDS', 'B_KO_ODDS']] = test.apply(get_odds, axis=1)
    return test


def remove_nans(fights_df):
    fights_df['R_AGE'].replace(['', 0], np.nan, inplace=True)
    fights_df['R_HEIGHT'].replace(['', 0], np.nan, inplace=True)
    # fights_df['R_WEIGHT'].replace(['', 0], np.nan, inplace=True)
    fights_df['R_REACH'].replace(['', 0], np.nan, inplace=True)
    fights_df['B_AGE'].replace(['', 0], np.nan, inplace=True)
    fights_df['B_HEIGHT'].replace(['', 0], np.nan, inplace=True)
    # fights_df['B_WEIGHT'].replace(['', 0], np.nan, inplace=True)
    fights_df['B_REACH'].replace(['', 0], np.nan, inplace=True)
    fights_df.dropna(subset=['R_AGE', 'R_HEIGHT', 'R_REACH', 'B_AGE', 'B_HEIGHT', 'B_REACH',
                             'R_ODDS', 'B_ODDS'], inplace=True)

    return fights_df


def rename_columns_finale(fights_df):
    mapping = {
        'R_fighter': 'R_NAME',
        'B_fighter': 'B_NAME',
        'win_by': 'WIN_BY',
        'last_round': 'LAST_ROUND',
        'last_round_time': 'LAST_ROUND_TIME',
        'Referee': 'REFEREE',
        'date': 'DATE',
        'location': 'LOCATION',
        'Winner': 'WINNER'
    }
    fights_df.rename(columns=mapping, inplace=True)
    return fights_df


def change_column_order(fights_df):
    return fights_df.reindex(columns=target_stats)


pd.options.mode.chained_assignment = None
locale.setlocale(locale.LC_ALL, 'en_US.utf8')
fights_total_df, fighters_total_df, odds_total_df = read_files()
fights_total_df = drop_draws_and_old_fights(fights_total_df)
fights_total_df = convert_times_to_seconds(fights_total_df)
fights_total_df = time_to_total(fights_total_df)
fights_total_df = create_title_bout_feature(fights_total_df)
fights_total_df = create_weight_category(fights_total_df)
fights_total_df = edit_win_by(fights_total_df)
fights_total_df = create_num_of_rounds(fights_total_df)
fights_total_df = edit_strikes(fights_total_df)
fights_total_df = drop_redundant_columns(fights_total_df)
fights_total_df = add_fighters_details(fights_total_df, fighters_total_df)
odds_total_df = reformat_odds_date(odds_total_df)
fights_total_df = add_odds(fights_total_df, odds_total_df)
fights_total_df = remove_nans(fights_total_df)
fights_total_df = rename_columns_finale(fights_total_df)
# fights_total_df = change_column_order(fights_total_df)

for col in fights_total_df.columns:
    print(col)

current_file = os.path.abspath(os.path.dirname(__file__))
# filename = os.path.join(current_file, '../../../data/preprocessed/cleaned_fights_data_no_nans_2022.csv')
filename = os.path.join(current_file, '../../../data/preprocessed/cleaned_fights_data.csv')

##TODO FIX AGE, WEIGHT


print(fights_total_df)
fights_total_df.to_csv(filename, index=False)


