"""
Describes the paths of data files. Implemented for usage in src/run/*
"""
import os
current_file = os.path.abspath(os.path.dirname(__file__))

# PER_MIN = os.path.join(current_file, '../../data/features/avgs_per_minute/per_min.csv')

PER_MIN_NO_DEBUTS = os.path.join(current_file, '../../data/features/avgs_per_minute/per_min_no_debuts.csv')
PER_MIN = os.path.join(current_file, '../../data/features/avgs_per_minute/per_min_debuts.csv')

PER_MIN_WEIGHTED = os.path.join(current_file,
                                '../../data/features/avgs_per_minute/per_min_weighted_debuts.csv')

PER_MIN_WEIGHTED_NO_DEBUTS = os.path.join(current_file,
                                '../../data/features/avgs_per_minute/per_min_weighted_no_debuts.csv')

PER_MIN_WEIGHTED_NO_DEBUTS_OPPOSITION = os.path.join(current_file,
                                '../../data/features/avgs_per_minute/per_min_weighted_no_debuts_opposition.csv')

