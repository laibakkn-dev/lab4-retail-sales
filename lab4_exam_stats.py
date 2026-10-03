import pandas as pd
import numpy as np
from scipy import stats

scores = pd.Series([72, 65, 88, np.nan, 54, 91, 76, np.nan, 83, 69])

print("Descriptive Statistics:")
print(scores.describe())

clean_scores = scores.dropna()

print("\nSkewness:", stats.skew(clean_scores))
print("Kurtosis:", stats.kurtosis(clean_scores))

print("\nMissing values:", scores.isnull().sum())

mean_score = scores.mean()
scores = scores.fillna(mean_score)

print("\nFilled scores:")
print(scores)