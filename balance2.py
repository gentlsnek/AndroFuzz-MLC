import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('merged_cic_andmal2017_risky.csv')

# Drop duplicates and handle missing values
df = df.drop_duplicates().dropna()

# Correct label for malware_2015_1016 → reassign as benign (label 0)
df['label'] = df['label'].replace('malware_2015_1016', 'benign')

# Drop non-numeric columns (like file names, if present)
if 'filename' in df.columns:
    df = df.drop(columns=['filename'])

# Convert categorical features to numerical
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df[categorical_cols] = df[categorical_cols].astype('category').apply(lambda x: x.cat.codes)

# Print current class distribution
print("Original class distribution:\n", df['label'].value_counts())

# Ensure labels are numerically mapped: benign=0, malware=1, risky=2
# Reclassify original label values into 0: benign, 1: malware, 2: risky
label_map = {
    1: 0,  # benign
    5: 1,  # malware
    4: 1,
    2: 1,
    0: 1,
    3: 2   # risky
}
df['label'] = df['label'].map(label_map)

# Check for any unmapped values
if df['label'].isnull().any():
    print("Warning: Found NaNs in label column after mapping!")

# Drop rows with NaN labels just in case
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)  # Ensure it's int for sampling

# Separate features and target
X = df.drop(columns=['label'])
y = df['label']

# Random undersampling (only benign and malware)
undersample_strategy = {
    0: 300,  # benign
    1: 300   # malware
    # Don't include risky (label 2) — we keep all of them
}
rus = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Append all original risky (label 2) samples back in
risky_df = df[df['label'] == 2]
X_risky = risky_df.drop(columns=['label'])
y_risky = risky_df['label']

X_combined = pd.concat([X_resampled, X_risky], ignore_index=True)
y_combined = pd.concat([y_resampled, y_risky], ignore_index=True)

# Apply SMOTE to boost risky samples to 100
smote = SMOTE(sampling_strategy={2: 100}, random_state=42)
X_final, y_final = smote.fit_resample(X_combined, y_combined)

# Convert back to DataFrame
df_final = pd.DataFrame(X_final, columns=X.columns)
df_final['label'] = y_final

# Print final distribution
print("\nFinal class distribution:\n", df_final['label'].value_counts())

# Save to CSV
df_final.to_csv('balanced_dataset_risky.csv', index=False)
