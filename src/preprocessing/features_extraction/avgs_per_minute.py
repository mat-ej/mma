import math
import numpy as np
import pandas as pd
import os
import locale

from src.preprocessing.features_extraction.feature_variables import (  # isort:skip
    copied_stats,
    red_fighter,
    blue_fighter,
    red_target_stats,
    blue_target_stats,
    red_stats,
    blue_stats,
    manipulation_columns,
    red_mapping,
    blue_mapping,
    red_stats_incl_absorbed,
    blue_stats_incl_absorbed,
    red_stats_opposition,
    blue_stats_opposition,
    red_extra_features,
    blue_extra_features
)

from src.services.paths import (
    PER_MIN,
    PER_MIN_NO_DEBUTS,
    PER_MIN_WEIGHTED,
    PER_MIN_WEIGHTED_NO_DEBUTS,
    PER_MIN_WEIGHTED_NO_DEBUTS_OPPOSITION
)


def copy_red(x):
    copy = x[red_stats + blue_stats + ['TOTAL_FIGHT_TIME', 'DATE']].copy()
    copy.rename(columns=red_mapping, inplace=True)
    copy['WIN'] = x['WINNER'] == x['R_NAME']
    return copy


def copy_blue(x):
    copy = x[red_stats + blue_stats + ['TOTAL_FIGHT_TIME', 'DATE']].copy()
    copy.rename(columns=blue_mapping, inplace=True)
    copy['WIN'] = x['WINNER'] == x['B_NAME']
    return copy


class AverageFeatures:
    def __init__(self, per, include_opposition=False, debuts=False, time_weighting=False, style_weighting=False, discount_rate=0.5):
        self.fights = None
        self.output = None
        self.output_list = []
        self.per = per
        self.include_opposition = include_opposition
        self.debuts = debuts
        self.time_weighting = time_weighting
        self.style_weighting = style_weighting
        self.discount_rate = discount_rate

    def get_fights(self):
        current_file = os.path.abspath(os.path.dirname(__file__))
        filename_fights = os.path.join(current_file,
                                       '/home/m/repo/ml-in-combat-sports/data/preprocessed/cleaned_fights_data_one_hot.csv')
        try:
            self.fights = pd.read_csv(filename_fights, sep=',')
            self.output = pd.DataFrame(
                columns=copied_stats + red_fighter + red_extra_features + red_target_stats + blue_fighter + blue_extra_features + blue_target_stats)
            # self.output = self.fights[copied_stats]
            print()
        except Exception as e:
            raise FileNotFoundError("Cannot find the source file:"
                                    '/home/m/repo/ml-in-combat-sports/data/preprocessed/cleaned_fights_data_one_hot.csv')

    def get_previous_fights(self, name, start_index):
        previous_fights = self.fights.iloc[(start_index + 1):]
        fights_as_red = previous_fights.loc[previous_fights['R_NAME'] == name]
        fights_as_blue = previous_fights.loc[previous_fights['B_NAME'] == name]
        all_fights = pd.concat([copy_red(fights_as_red), copy_blue(fights_as_blue)])
        # all_fights = all_fights.sort_values(by='DATE')
        return all_fights

    def get_averages_per_minute(self, all_fights, colour, current_date, time_weighting=False):
        num_of_fights = all_fights.shape[0]
        if num_of_fights == 0:
            if self.debuts:
                zeros = np.zeros(shape=(len(manipulation_columns) + 1,))
                zeros[-1] = 1
                if colour == 'Red':
                    return pd.Series(zeros, index=red_stats_incl_absorbed)
                else:
                    return pd.Series(zeros, index=blue_stats_incl_absorbed)
            else:
                return None
        averages = np.zeros(shape=(len(manipulation_columns) + 1,), dtype=float)
        bout_lengths = all_fights['TOTAL_FIGHT_TIME'].to_numpy() / 60.0
        weights = None
        if time_weighting:
            dates = pd.to_datetime(all_fights['DATE'], format='%B %d, %Y')
            year_delta = ((current_date - dates).astype('timedelta64[D]') / 365.25).astype(float)
            baseline = np.full(shape=(num_of_fights,), fill_value=self.discount_rate)
            weights = np.power(baseline, year_delta.values)
            # print(weights)
            if np.any(weights < 0):
                print('Negative weight')
                return -1
        else:
            weights = np.ones(shape=(num_of_fights,))

        for i in range(len(manipulation_columns)):
            print(manipulation_columns[i])
            averages[i] = (all_fights[manipulation_columns[i]] * weights).sum() / (bout_lengths * weights).sum()
        if colour == 'Red':
            return pd.Series(averages, index=red_stats_incl_absorbed)
        else:
            return pd.Series(averages, index=blue_stats_incl_absorbed)

    def save(self):
        filename = None
        if self.debuts and not self.time_weighting and not self.include_opposition:
            filename = PER_MIN
        if not self.debuts and not self.time_weighting and not self.include_opposition:
            filename = PER_MIN_NO_DEBUTS
        if self.debuts and self.time_weighting and not self.include_opposition:
            filename = PER_MIN_WEIGHTED
        if not self.debuts and self.time_weighting and not self.include_opposition:
            filename = PER_MIN_WEIGHTED_NO_DEBUTS
        if not self.debuts and self.time_weighting and self.include_opposition:
            filename = PER_MIN_WEIGHTED_NO_DEBUTS_OPPOSITION
        print(filename)
        self.output.to_csv(filename, index=False)

    def get_output(self):
        return self.output

    def produce_features(self):
        def edit_booleans(x):
            if x['WINNER'] == x['R_NAME']:
                x['WINNER'] = 1
            elif x['WINNER'] == x['B_NAME']:
                x['WINNER'] = 0

            if x['TITLE_BOUT']:
                x['TITLE_BOUT'] = 1
            elif not x['TITLE_BOUT']:
                x['TITLE_BOUT'] = 0

        def produce_extra_features(x, all_fights, colour):
            fights = all_fights.sort_index()
            wins = all_fights['WIN'].sum()
            win_streak = 0
            loss_streak = 0
            num_of_fights = fights.shape[0]
            for index, fight in fights.iterrows():
                if fight['WIN']:
                    win_streak += 1
                else:
                    break
            for index, fight in fights.iterrows():
                if not fight['WIN']:
                    loss_streak += 1
                else:
                    break
            if colour == 'Red':
                x['R_WIN_PCT'] = float(wins / num_of_fights)
                x['R_WIN_STREAK'] = win_streak
                x['R_LOSS_STREAK'] = loss_streak
            else:
                x['B_WIN_PCT'] = float(wins / fights.shape[0])
                x['B_WIN_STREAK'] = win_streak
                x['B_LOSS_STREAK'] = loss_streak

        def produce_row(x, index):
            r_name = x['R_NAME']
            b_name = x['B_NAME']
            copy = x[copied_stats + red_fighter + blue_fighter].copy()
            fight_date = pd.to_datetime(x['DATE'], format='%B %d, %Y')
            edit_booleans(copy)

            red_fights = self.get_previous_fights(r_name, index)
            blue_fights = self.get_previous_fights(b_name, index)

            red_averages, blue_averages = None, None

            if self.per == 'minute':
                red_averages = self.get_averages_per_minute(red_fights, 'Red', fight_date, self.time_weighting)
                blue_averages = self.get_averages_per_minute(blue_fights, 'Blue', fight_date, self.time_weighting)

            # for debuts excluded - leave the fight out
            if red_averages is None or blue_averages is None:
                return

            produce_extra_features(copy, red_fights, 'Red')
            produce_extra_features(copy, blue_fights, 'Blue')

            copy = copy.append(red_averages)
            copy = copy.append(blue_averages)

            self.output_list.append(copy)
            self.output = self.output.append(copy, ignore_index=True)

        self.get_fights()
        for ind, fight in self.fights.iterrows():
            print(ind)
            print(str(ind) + ' / ' + str(self.fights.shape[0]))
            produce_row(fight, ind)

        self.output = self.output.reindex(
            columns=copied_stats + red_fighter + red_extra_features + red_stats_incl_absorbed + blue_fighter +
                    blue_extra_features + blue_stats_incl_absorbed)
        if not self.include_opposition:
            self.output = self.output.drop(red_stats_opposition + blue_stats_opposition, axis=1)
