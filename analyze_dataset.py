import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

# fetch dataset
mushroom = fetch_ucirepo(id=73)

# data (as pandas dataframes)
X = mushroom.data.features
y = mushroom.data.targets

print("Dataset shape:", X.shape)
print("Target shape:", y.shape)
print("\nFirst few rows of features:")
print(X.head())
print("\nFirst few rows of target:")
print(y.head())

print("\nFeature names:")
print(X.columns.tolist())
print("\nTarget name:", y.columns.tolist())

print("\nData types:")
print(X.dtypes)

print("\nMissing values per feature:")
missing = X.isnull().sum()
print(missing[missing > 0])

print("\nUnique values per feature (categorical):")
for col in X.columns:
    uniq = X[col].nunique()
    print(f"{col}: {uniq} unique values")

print("\nTarget distribution:")
print(y.iloc[:, 0].value_counts())