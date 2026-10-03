import pandas as pd
import numpy as np
from scipy.stats import zscore, skew, kurtosis

data = {
    "Hour": range(1, 16),
    "Temperature": [
        22, 23, np.nan, np.nan, np.nan,
        25, 26, 58, 24, 23,
        22, 21, 23, 24, 25
    ]
}

df = pd.DataFrame(data)

# Statistics BEFORE cleaning
print("Stats BEFORE cleaning:")
print("Mean:", df["Temperature"].mean())
print("Std:", df["Temperature"].std())
print("Skewness:", skew(df["Temperature"].dropna()))
print("Kurtosis:", kurtosis(df["Temperature"].dropna()))

# Save raw temperature
df["Temp_raw"] = df["Temperature"]

# Linear interpolation
df["Temperature"] = df["Temperature"].interpolate(method="linear")

print("\nAfter interpolation, missing values:",
      df["Temperature"].isnull().sum())

# Z-score for outlier detection
z_scores = zscore(df["Temperature"])

# Detect outliers using threshold 2
outlier_mask = abs(z_scores) > 2

print("\nOutlier detected:")
print(df[outlier_mask][["Hour", "Temperature"]])

# Replace outlier with median
median_temperature = df["Temperature"].median()
df.loc[outlier_mask, "Temperature"] = median_temperature

print("\nReplaced with column median:", median_temperature)

# Standardization after cleaning
df["Temp_Zscore"] = zscore(df["Temperature"])

# Before/After comparison
print("\nBefore/After comparison:")
print(df[["Hour", "Temp_raw", "Temperature", "Temp_Zscore"]])

# Statistics AFTER cleaning
print("\nStats AFTER cleaning:")
print("Mean:", df["Temperature"].mean())
print("Std:", df["Temperature"].std())
