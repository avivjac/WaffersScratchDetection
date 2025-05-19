import pandas as pd
import zipfile
from datetime import datetime
#Import sklearn relavant libraries
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage import convolve

#load zip file
zf = zipfile.ZipFile('data.zip') 

#load train data
df_wafers = pd.read_csv(zf.open('wafers_train.csv'))
df_wafers.head()

#load test data
df_wafers_test = pd.read_csv(zf.open('wafers_test.csv'))
df_wafers_test.head()

def plot_wafer_maps(wafer_df_list, figsize, labels = True):
    """
    plot wafer maps for list of df of wafers

    :param wafer_df_list: list, The list of df's of the wafers
    :param figsize: int, the size of the figsize height 
    :param labels: bool, Whether to show the layer of labels (based on column 'IsScratchDie')
    
    :return: None
    """
    def plot_wafer_map(wafer_df, ax, map_type):
        wafer_size = len(wafer_df)
        s = 2**17/(wafer_size)
        if map_type == 'Label':
            mes = 'Scratch Wafer' if (wafer_df['IsScratchDie'] == True).sum()>0 else 'Non-Scratch Wafer'
        else:
            mes = 'Yield: ' + str(round((wafer_df['IsGoodDie']).sum()/(wafer_df['IsGoodDie']).count(), 2)) 
        
        ax.set_title(f'{map_type} | Wafer Name: {wafer_df["WaferName"].iloc[0]}, \nSum: {len(wafer_df)} dies. {mes}', fontsize=20)
        ax.scatter(wafer_df['DieX'], wafer_df['DieY'], color = 'green', marker='s', s = s)

        bad_bins = wafer_df.loc[wafer_df['IsGoodDie'] == False]
        ax.scatter(bad_bins['DieX'], bad_bins['DieY'], color = 'red', marker='s', s = s)
        
        if map_type == 'Label':
            scratch_bins = wafer_df.loc[(wafer_df['IsScratchDie'] == True) & (wafer_df['IsGoodDie'] == False)]
            ax.scatter(scratch_bins['DieX'], scratch_bins['DieY'], color = 'blue', marker='s', s = s)

            ink_bins = wafer_df.loc[(wafer_df['IsScratchDie'] == True) & (wafer_df['IsGoodDie'] == True)]
            ax.scatter(ink_bins['DieX'], ink_bins['DieY'], color = 'yellow', marker='s', s = s)

            ax.legend(['Good Die', 'Bad Die', 'Scratch Die', 'Ink Die'], fontsize=8)
        else:
            ax.legend(['Good Die', 'Bad Die'], fontsize=8)

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False) 
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    if labels:
        fig, ax = plt.subplots(2, len(wafer_df_list), figsize=(figsize*len(wafer_df_list), figsize*2))
        for idx1, wafer_df in enumerate(wafer_df_list):
            for idx2, map_type in enumerate(['Input', 'Label']):
                plot_wafer_map(wafer_df, ax[idx2][idx1], map_type)
    else:
        fig, ax = plt.subplots(1, len(wafer_df_list), figsize=(figsize*len(wafer_df_list), figsize))
        for idx, wafer_df in enumerate(wafer_df_list):
            plot_wafer_map(wafer_df, ax[idx], 'Input')

    plt.show()

# Function to add the following features: neighbor in 3X3, distance from center, angle of bad neighbors, local density, and cluster level
def add_neighbor_features_3x3(df, label_col='IsGoodDie'):
    df = df.copy()
    all_augmented = []

    # 4x4 kernel
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0  # the center of the matrix

    # Sobel kernels for gradient calculation
    gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    for wafer, wafer_df in df.groupby("WaferName"):
        x_coords = wafer_df['DieX'].astype(int)
        y_coords = wafer_df['DieY'].astype(int)

        x_min, y_min = x_coords.min(), y_coords.min()
        x_max, y_max = x_coords.max(), y_coords.max()

        grid_w, grid_h = x_max - x_min + 1, y_max - y_min + 1
        grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
        index_map = {}

        for i, row in wafer_df.iterrows():
            x = int(row['DieX']) - x_min
            y = int(row['DieY']) - y_min
            grid[y, x] = 1 if not row[label_col] else 0
            index_map[(x, y)] = i

        bad_neighbors = convolve(grid, kernel, mode='constant', cval=0)
        grad_x = convolve(grid, gx, mode='constant', cval=0)
        grad_y = convolve(grid, gy, mode='constant', cval=0)
        angles = np.arctan2(grad_y, grad_x)

        wafer_df['bad_neighbors_count_3x3'] = 0
        wafer_df['angle_of_bad_neighbors'] = 0.0

        for (x, y), idx in index_map.items():
            wafer_df.at[idx, 'bad_neighbors_count_3x3'] = bad_neighbors[y, x]
            wafer_df.at[idx, 'angle_of_bad_neighbors'] = angles[y, x]

        # Calculate distance from center
        cx = wafer_df['DieX'].mean()
        cy = wafer_df['DieY'].mean()
        wafer_df['dist_from_center'] = np.sqrt((wafer_df['DieX'] - cx)**2 + (wafer_df['DieY'] - cy)**2)

        all_augmented.append(wafer_df)

    result = pd.concat(all_augmented, ignore_index=True)

    # sin/cos
    result['angle_sin'] = np.sin(result['angle_of_bad_neighbors'])
    result['angle_cos'] = np.cos(result['angle_of_bad_neighbors'])

    # Calculate local density
    result['local_density'] = result['bad_neighbors_count_3x3'] / 8.0

    # Calculate cluster level
    def cluster_level(n):
        if n <= 1:
            return 0
        elif n <= 3:
            return 1
        else:
            return 2

    result['cluster_level'] = result['bad_neighbors_count_3x3'].apply(cluster_level)

    return result

# Add neighbor features to the training set
df_wafers_aug = add_neighbor_features_3x3(df_wafers)

# Split the data into features and target variable
features = ['DieX', 'DieY', 'IsGoodDie', 'bad_neighbors_count_3x3', 'angle_sin', 'angle_cos', 'cluster_level', 'dist_from_center', 'local_density']
X = df_wafers_aug[features]
y = df_wafers_aug['IsScratchDie']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define he model using XGBooster 
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=129 # 129:1 ratio of positive to negative samples
)
model.fit(X_train, y_train)

# validation
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred_test = (y_pred_proba > 0.9).astype(int)

# calculae accuracy on validation set 
acc = accuracy_score(y_test, y_pred_test)
print(f"Accuracy on Validation set: {acc:.4f}\n")

print("Classification Report on Validation set:")
print(classification_report(y_test, y_pred_test))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Validation Set')
plt.show()

# add neighbor features to test set
df_wafers_test_aug = add_neighbor_features_3x3(df_wafers_test)

# make predictions on test set
X_test = df_wafers_test_aug[features]
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred_test = (y_pred_proba > 0.9).astype(int)
df_wafers_test_aug['IsScratchDie'] = y_pred_test

# draw waffers maps - test set with predictions
n_samples = 4
list_sample_test = [df_wafers_test_aug.groupby('WaferName').get_group(group) for group in df_wafers_test_aug['WaferName'].value_counts().sample(n_samples, random_state=20).index]
plot_wafer_maps(list_sample_test, figsize = 8, labels = True)

df_wafers_test = df_wafers_test_aug[['WaferName', 'DieX', 'DieY', 'IsGoodDie', 'IsScratchDie']]