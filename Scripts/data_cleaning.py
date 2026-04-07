import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("Datasets/synthetic_micro_doppler_dataset.csv")
print(f"Original Dataset Shape: {df.shape}")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 2. DATA CLEANING
print("\n--- Data Cleaning ")
missing_values = X.isnull().sum().sum()
print(f"Total missing values found: {missing_values}")

if missing_values > 0:
    X = X.ffill(axis=1).bfill(axis=1)
    print("Missing values have been filled using forward and backward fill.")

scaler = StandardScaler()
X_cleaned = pd.DataFrame(scaler.fit_transform(X.T).T, columns=X.columns)

df_clean = pd.concat([X_cleaned, y.rename('label')], axis=1)


print("\n Generating EDA Visualization")

plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df_clean, palette='Set2')
plt.title('Class Distribution (0: Bird, 1: Drone)')
plt.xlabel('Target Class')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(14, 5))
sample_bird = df_clean[df_clean['label'] == 0].iloc[0, :-1].values[:500]
sample_drone = df_clean[df_clean['label'] == 1].iloc[0, :-1].values[:500]

plt.plot(sample_bird, label='Bird (Class 0)', color='blue', alpha=0.7)
plt.plot(sample_drone, label='Drone (Class 1)', color='orange', alpha=0.7)
plt.title('Cleaned Radar Signal Comparison (First 500 time steps)')
plt.xlabel('Time Step')
plt.ylabel('Normalized Amplitude')
plt.legend()
plt.tight_layout()
plt.show()