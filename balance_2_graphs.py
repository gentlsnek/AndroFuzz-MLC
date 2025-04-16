import pandas as pd
import matplotlib.pyplot as plt
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

# Print and store original class distribution
initial_distribution = df['label'].value_counts()
print("Original class distribution:\n", initial_distribution)

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

# Drop rows with NaN labels
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# Separate features and target
X = df.drop(columns=['label'])
y = df['label']

# Random undersampling (only benign and malware)
undersample_strategy = {
    0: 300,  # benign
    1: 300   # malware
}
rus = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Add back risky samples
risky_df = df[df['label'] == 2]
X_risky = risky_df.drop(columns=['label'])
y_risky = risky_df['label']

X_combined = pd.concat([X_resampled, X_risky], ignore_index=True)
y_combined = pd.concat([y_resampled, y_risky], ignore_index=True)

# Store post-reduction distribution
after_reduction_distribution = y_combined.value_counts()
print("\nAfter reduction class distribution:\n", after_reduction_distribution)

# Apply SMOTE to boost risky class to 100 samples
smote = SMOTE(sampling_strategy={2: 100}, random_state=42)
X_final, y_final = smote.fit_resample(X_combined, y_combined)

# Store final distribution
final_distribution = pd.Series(y_final).value_counts()
print("\nFinal class distribution:\n", final_distribution)

# Save to CSV
df_final = pd.DataFrame(X_final, columns=X.columns)
df_final['label'] = y_final
#df_final.to_csv('balanced_dataset_risky.csv', index=False)

# ---- PLOTTING ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Define category labels for plotting
category_labels = {0: 'benign', 1: 'malware', 2: 'risky'}

# 1. Initial distribution (no category names, just numeric labels)
x0 = initial_distribution.index.astype(int)
axes[0].bar(x0, initial_distribution.values, color='skyblue')
axes[0].set_title("Initial Class Distribution")
axes[0].set_xlabel("Label")
axes[0].set_ylabel("Count")

# 2. After reduction (with category names)
x1 = after_reduction_distribution.index.astype(int)
axes[1].bar([category_labels.get(i, str(i)) for i in x1], after_reduction_distribution.values, color='orange')
axes[1].set_title("After Undersampling (Risky Preserved)")
axes[1].set_xlabel("Category")

# 3. After SMOTE (with category names)
x2 = final_distribution.index.astype(int)
axes[2].bar([category_labels.get(i, str(i)) for i in x2], final_distribution.values, color='green')
axes[2].set_title("After SMOTE Applied")
axes[2].set_xlabel("Category")

plt.tight_layout()
plt.show()
