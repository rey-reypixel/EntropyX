import pandas as pd

df = pd.read_csv("datasets/ransomware_dataset.csv")

# Use first row as header
df.columns = df.iloc[0]

# Remove header row from data
df = df[1:]

# Reset index
df = df.reset_index(drop=True)

print("Shape:", df.shape)
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print(df["family"].value_counts())

print("\nClass Distribution:")
print(df["family"].value_counts())

print("\nPercentage Distribution:")
print(df["family"].value_counts(normalize=True) * 100)

print("\nMissing Values:")
missing = df.isnull().sum()

print(missing[missing > 0])

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

print("\nData Types:")
print(df.dtypes)

print(df.describe())

with open("eda_report.txt", "w") as f:
    f.write(f"Shape: {df.shape}\n\n")
    f.write("Class Distribution:\n")
    f.write(str(df["family"].value_counts()))