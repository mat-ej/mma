from src.preprocessing.features_extraction.avgs_per_minute import AverageFeatures

#Averages = AverageFeatures(per='minute', include_opposition=False, debuts=False, time_weighting=False, style_weighting=False)
#Averages.produce_features()

WeightedAverages = AverageFeatures(per='minute', include_opposition=True, debuts=False, time_weighting=True, style_weighting=False, discount_rate=0.4)
WeightedAverages.produce_features()
WeightedAverages.save()

