# California House Price Prediction

A machine learning project that predicts California house prices using Python, Pandas, NumPy, Scikit-learn, and Random Forest Regression.

## Creator

Nosad
Computer Science Graduate

## Project Overview

This project implements an end-to-end machine learning workflow for predicting California house prices.

The target column is:

`median_house_value`

Since the target is a numerical value, the project uses `RandomForestRegressor`.

The complete workflow is:

```text
housing.csv
    ↓
Stratified Train/Test Split
    ↓
train.csv + test.csv
    ↓
Feature / Label Separation
    ↓
Data Preprocessing
    ↓
Random Forest Regression
    ↓
Hyperparameter Tuning
    ↓
Save Model + Pipeline
    ↓
input.csv
    ↓
Predictions
    ↓
Actual vs Predicted Comparison
    ↓
Mean Absolute Error
    ↓
output.csv
```

## Dataset

The project uses California housing data containing numerical and categorical features.

Main features include:

* longitude
* latitude
* housing_median_age
* total_rooms
* total_bedrooms
* population
* households
* median_income
* ocean_proximity

Target:

* median_house_value

The original dataset is stored in:

```text
data/housing.csv
```

## Stratified Train/Test Split

The project uses `StratifiedShuffleSplit` instead of a simple random split.

The `median_income` column is divided into five income categories:

```text
0.0 - 1.5
1.5 - 3.0
3.0 - 4.5
4.5 - 6.0
6.0+
```

The split maintains a similar distribution of these income categories in both the training and testing datasets.

The temporary `income_cat` column is removed after the split because it is only required for creating the stratified split.

## Data Preprocessing

Scikit-learn pipelines are used to handle preprocessing.

### Numerical Features

Numerical features go through:

1. Missing-value handling using median imputation
2. Standard scaling using `StandardScaler`

### Categorical Features

The `ocean_proximity` column is categorical.

It is converted into numerical form using `OneHotEncoder`.

The encoder uses:

```python
handle_unknown="ignore"
```

This prevents errors when prediction data contains an unknown category.

## Machine Learning Model

The project uses:

```text
RandomForestRegressor
```

Random Forest is suitable for this problem because `median_house_value` is a continuous numerical target.

The model combines predictions from multiple decision trees to produce the final house price prediction.

## Hyperparameter Tuning

`GridSearchCV` is used to find a better combination of Random Forest hyperparameters.

The project tunes:

* `n_estimators`
* `max_depth`
* `min_samples_split`

The search is intentionally kept small to reduce computational requirements.

The project uses:

```text
2-fold cross-validation
```

The best parameters found during development were:

```text
n_estimators = 200
max_depth = 30
min_samples_split = 2
```

## Preventing Preprocessing Leakage

The preprocessing pipeline and Random Forest model are combined into a single Scikit-learn pipeline before hyperparameter tuning.

The structure is:

```text
Input Data
    ↓
Preprocessing
    ↓
Random Forest
```

`GridSearchCV` evaluates this complete pipeline during each cross-validation fold.

This ensures that preprocessing is fitted separately within each fold instead of using information from the validation data.

## Model Evaluation

The project uses Mean Absolute Error (MAE) to evaluate predictions.

MAE measures the average absolute difference between the actual house values and the predicted house values.

### Cross-Validation MAE

```text
34,236.30
```

### Final Test MAE

```text
30,921.70
```

The final test MAE means the model's predictions differed from the actual house values by approximately `$30,922` on average on the unseen test dataset.

## Input Data

`input.csv` contains the house features that are given to the trained model for prediction.

It must not contain:

```text
median_house_value
```

because this is the target being predicted.

The sample input file is located at:

```text
data/input.csv
```

## Testing and Output

The project keeps the actual test values in `test.csv`.

During testing:

```text
test.csv
    ↓
Actual median_house_value

input.csv
    ↓
Trained Model
    ↓
Predicted Values
```

The actual and predicted values are then combined into:

```text
output.csv
```

The output contains:

```text
median_house_value
Predicted Median House Value
```

along with the original input features.

This makes it possible to compare the actual and predicted values row by row.

## First Run vs Later Runs

The program checks whether `model.pkl` exists.

### First Run

If `model.pkl` does not exist:

```text
housing.csv
    ↓
Create train.csv and test.csv
    ↓
Preprocess training data
    ↓
Train Random Forest
    ↓
Hyperparameter tuning
    ↓
Save model.pkl
    ↓
Save pipeline.pkl
```

### Later Runs

If `model.pkl` already exists, the training phase is skipped.

The program directly performs testing:

```text
model.pkl + pipeline.pkl
        ↓
data/input.csv
        ↓
Predictions
        ↓
MAE
        ↓
output.csv
```

To retrain and tune the model again, delete `model.pkl` and `pipeline.pkl` and run the program again.

## Generated Files

The following files are generated automatically:

```text
train.csv
test.csv
model.pkl
pipeline.pkl
output.csv
```

These files are excluded from GitHub using `.gitignore`.

### model.pkl

Stores the trained Random Forest model.

### pipeline.pkl

Stores the fitted preprocessing pipeline.

### train.csv

Contains the training portion created from `housing.csv`.

### test.csv

Contains the testing portion created from `housing.csv`.

### output.csv

Contains the actual and predicted house values.

## Project Structure

```text
california-house-price-prediction/
│
├── data/
│   ├── housing.csv
│   └── input.csv
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

Generated files are ignored by Git.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

## Main Scikit-learn Components

* `StratifiedShuffleSplit`
* `Pipeline`
* `ColumnTransformer`
* `SimpleImputer`
* `StandardScaler`
* `OneHotEncoder`
* `RandomForestRegressor`
* `GridSearchCV`
* `mean_absolute_error`

## How to Run

### 1. Clone the repository

```bash
git clone <repository-url>
```

### 2. Open the project folder

```bash
cd california-house-price-prediction
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the program

```bash
python main.py
```

On the first run, the program creates the train/test datasets, trains the model, performs hyperparameter tuning, and saves the trained model and preprocessing pipeline.

On later runs, the saved model is used to make predictions from `data/input.csv`.

The predictions are saved to:

```text
output.csv
```

## Important Notes

* Keep `data/input.csv` in the same feature format as the training data.
* Do not add `median_house_value` to `input.csv`.
* When performing final testing, `input.csv` must contain the same rows and order as the corresponding rows in `test.csv`.
* `model.pkl` and `pipeline.pkl` are generated automatically.
* The test dataset is kept separate from model training and hyperparameter tuning.
* The final MAE is calculated using the unseen test data.

## Future Improvements

Possible future improvements include:

* Feature importance analysis
* More extensive hyperparameter tuning
* Comparison with other regression models
* Prediction visualizations
* A Streamlit web interface
* Improved input validation

---

