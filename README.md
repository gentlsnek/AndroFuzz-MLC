# AndroFuzz - MultiLevel Classification

AndroFuzz is a machine learning pipeline for Android APK analysis and classification. It preprocesses APK files, extracts features, balances datasets, and trains models to classify APKs into categories such as benign, malware, and risky apps.

## Features

- **APK Feature Extraction:** Uses Androguard to extract static features from APK files.
- **Dataset Processing:** Cleans, balances, and merges datasets from multiple sources.
- **Model Training:** Supports multi-class classification using PyTorch and scikit-learn.
- **Visualization:** Provides Jupyter notebooks for model development and evaluation.
- **Automation:** Scripts for batch processing and merging of datasets.

## Directory Structure

```
├── apkpreprocess.py         # Extracts features from APKs and saves to CSV
├── balance_2_graphs.py      # Balances datasets and visualizes class distributions
├── balance_code.py          # Additional dataset balancing utilities
├── balance2.py              # Alternative balancing script
├── cleaning.py              # Cleans and preprocesses CSV datasets
├── count.py                 # Counts and summarizes dataset statistics
├── fixdataset.py            # Fixes issues in datasets
├── merge.py                 # Merges multiple processed CSVs into a single dataset
├── model4_multiclass.ipynb  # Jupyter notebook for multiclass model training/evaluation
├── model4.ipynb             # Jupyter notebook for advanced model training/evaluation
├── modelv3.ipynb            # Earlier model development notebook
├── modelv3.2.ipynb          # Improved model development notebook
├── processed_data/          # Directory for processed CSV feature files
├── *.csv                    # Various intermediate and final datasets
├── generator_benign_to_malware.pth # Trained model weights
├── README.md                # Project documentation
```

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/AndroFuzz-MLC.git
    cd AndroFuzz-MLC
    ```

2. **Install dependencies:**
    - Python 3.8+
    - [Androguard](https://github.com/androguard/androguard)
    - pandas, numpy, scikit-learn, torch, matplotlib, seaborn, notebook

    ```sh
    pip install -r requirements.txt
    ```

## Usage

### 1. Extract Features from APKs

Edit the dataset paths in `apkpreprocess.py` and run:

```sh
python apkpreprocess.py
```

This will process APKs in the specified directories and output feature CSVs to `processed_data/`.

### 2. Clean and Balance Datasets

- Clean datasets:
    ```sh
    python cleaning.py
    ```
- Balance datasets:
    ```sh
    python balance_code.py
    ```

### 3. Merge Datasets

Combine multiple feature CSVs into a single dataset:

```sh
python merge.py
```

### 4. Model Training and Evaluation

Use the provided Jupyter notebooks for model training and evaluation:

```sh
jupyter notebook model4_multiclass.ipynb
```

## Notebooks

- `model4_multiclass.ipynb`: Multiclass model training, ROC/AUC analysis.
- `model4.ipynb`: Advanced model training, ROC/AUC analysis.
- `modelv3.2.ipynb`: Earlier model experiments.

## Data

- Place your APK files in the directories specified in `apkpreprocess.py`.
- Processed feature CSVs will be saved in `processed_data/`.
- Merged and balanced datasets are available as CSV files in the root directory.

## License

Specify your license here.

## Acknowledgements

- [Androguard](https://github.com/androguard/androguard)
- scikit-learn, PyTorch, pandas, matplotlib, seaborn

---
*For more details, see the code in each script and notebook.*
