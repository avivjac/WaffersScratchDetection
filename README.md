# Wafer Scratch Detection

## Project Overview

This project implements a machine learning pipeline to detect scratches on semiconductor wafers based on die-level inspection data. Using spatial neighbor features and an XGBoost classifier, the model learns patterns of defective dies and predicts whether a wafer contains scratches.

## Table of Contents

* [Installation](#installation)
* [Data](#data)
* [Feature Engineering](#feature-engineering)
* [Usage](#usage)
* [Model Training & Evaluation](#model-training--evaluation)
* [Visualization](#visualization)
* [Contributing](#contributing)

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/wafer-scratch-detection.git
   cd wafer-scratch-detection
   ```
2. **Create & activate a virtual environment** (recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

> **Note:** `requirements.txt` should include:
>
> ```text
> pandas
> numpy
> scipy
> scikit-learn
> xgboost
> matplotlib
> seaborn
> ```

## Data

The `data.zip` archive contains two CSV files:

* `wafers_train.csv`: Training set with columns: `WaferName`, `DieX`, `DieY`, `IsGoodDie`, `IsScratchDie`.
* `wafers_test.csv`: Test set without the `IsScratchDie` label.

Unzip or let the script load the archive directly:

```python
zf = zipfile.ZipFile('data.zip')
df_train = pd.read_csv(zf.open('wafers_train.csv'))
```

## Feature Engineering

Custom features are computed per die:

* **Spatial neighbors**: number of bad neighbors in a 3×3 kernel (`bad_neighbors_count_3x3`).
* **Gradient angles**: direction of defect clusters (`angle_of_bad_neighbors`, expanded to sine/cosine).
* **Distance from center**: Euclidean distance to wafer center (`dist_from_center`).
* **Local density**: ratio of bad neighbors (`local_density`).
* **Cluster level**: categorizes neighborhood severity (`cluster_level`).

All features are added via the `add_neighbor_features_3x3` function in `main.py`.

## Usage

Run the main script to train the model and evaluate on validation data:

```bash
python main.py
```

This will:

1. Load and preprocess data.
2. Engineer features.
3. Split into train/validation sets.
4. Train an `XGBClassifier` with tuned hyperparameters.
5. Print accuracy, classification report, and show a confusion matrix heatmap.
6. Generate wafer map visualizations with predictions on test samples.

## Model Training & Evaluation

* **Algorithm**: XGBoost classifier
* **Key parameters**:

  * `n_estimators=300`
  * `max_depth=6`
  * `learning_rate=0.05`
  * `subsample=0.8`, `colsample_bytree=0.8`
  * `scale_pos_weight=129` (to handle class imbalance)
* **Threshold**: 0.9 for positive scratch detection

Example validation results:

```text
Accuracy on Validation set: 0.95
Classification Report:
              precision    recall  f1-score   support

       0.0       ...
       1.0       ...
```

## Visualization

Use `plot_wafer_maps` to inspect sample wafers:

```python
from main import plot_wafer_maps
list_sample = [group for _, group in test_df.groupby('WaferName')][:4]
plot_wafer_maps(list_sample, figsize=8, labels=True)
```

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to fork the repository and submit a pull request.


