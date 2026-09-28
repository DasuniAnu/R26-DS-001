# scripts/fix_unified_labels.py
import pandas as pd

df = pd.read_csv("dataset/unified_dataset.csv")

# Normalise language to lowercase
df['language'] = df['language'].str.lower().str.strip()

# Normalise label to int
df['label'] = df['label'].astype(int)

# Save back
df.to_csv("dataset/unified_dataset.csv", index=False)

print(f"Fixed. Total rows: {len(df)}")
print(df['language'].value_counts())
print(df['label'].value_counts())