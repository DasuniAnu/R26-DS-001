import pandas as pd

df = pd.read_csv("dataset/ground_truth_partial_v2.csv")

print(df.head())

print("\nRows with valid timestamps:")
print((df['fake_end_sec'] > 0).sum())

print("\nTotal rows:")
print(len(df))