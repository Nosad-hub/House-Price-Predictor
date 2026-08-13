import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"


# Function to build the preprocessing pipeline
def build_pipeline(num_attribs, cat_attribs):

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    return full_pipeline


if not os.path.exists(MODEL_FILE):

    # ============================================================
    # TRAINING PHASE
    # ============================================================

    # Step 1: Load the complete housing dataset
    housing = pd.read_csv("data/housing.csv")


    # Step 2: Create income categories for stratified splitting
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )


    # Step 3: Split the complete dataset into training and testing data
    # using StratifiedShuffleSplit
    split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    for train_index, test_index in split.split(housing, housing["income_cat"]):

        train_data = housing.loc[train_index].copy()
        test_data = housing.loc[test_index].copy()


    # Step 4: Remove the temporary income_cat column
    # because it is only used for stratified splitting
    train_data = train_data.drop("income_cat", axis=1)
    test_data = test_data.drop("income_cat", axis=1)


    # Step 5: Save the training and testing data as separate CSV files
    train_data.to_csv(TRAIN_FILE, index=False)
    test_data.to_csv(TEST_FILE, index=False)

    print("Training and testing data created and saved.")


    # Step 6: Load the training data
    # All further training work will be performed on training data
    housing = pd.read_csv(TRAIN_FILE)


    # Step 7: Separate the label from the features
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)


    # Step 8: Identify numerical and categorical attributes
    num_attribs = housing_features.drop(
        "ocean_proximity",
        axis=1
    ).columns.tolist()

    cat_attribs = ["ocean_proximity"]


    # Step 9: Build the preprocessing pipeline
    pipeline = build_pipeline(
        num_attribs,
        cat_attribs
    )


    # Step 10: Create the Random Forest Regressor
    model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )


    # Step 11: Combine preprocessing and Random Forest into one pipeline
    # This prevents preprocessing information from leaking between CV folds
    full_model_pipeline = Pipeline([
        ("preprocessing", pipeline),
        ("model", model)
    ])


    # Step 12: Define a smaller set of hyperparameters
    # This keeps the tuning process faster for our computer
    param_grid = [
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [20, 30],
            "model__min_samples_split": [2]
        },
        {
            "model__n_estimators": [200],
            "model__max_depth": [None, 30],
            "model__min_samples_split": [5]
        }
    ]


    # Step 13: Perform hyperparameter tuning using GridSearchCV
    # Each combination will be evaluated using 2-fold cross-validation
    grid_search = GridSearchCV(
        full_model_pipeline,
        param_grid,
        cv=2,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )


    # Step 14: Find the best hyperparameters
    # Preprocessing is now performed separately inside each CV fold
    grid_search.fit(
        housing_features,
        housing_labels
    )


    # Step 15: Get the best trained complete pipeline
    best_pipeline = grid_search.best_estimator_


    # Step 16: Extract the best trained model
    model = best_pipeline.named_steps["model"]


    # Step 17: Extract the fitted preprocessing pipeline
    pipeline = best_pipeline.named_steps["preprocessing"]


    # Step 18: Display the best hyperparameters
    print("Best Hyperparameters:")
    print(grid_search.best_params_)


    # Step 19: Display the best cross-validation MAE
    best_mae = -grid_search.best_score_

    print(f"Best Cross-validation MAE: {best_mae:.2f}")


    # Step 20: Save the tuned model and fitted preprocessing pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Tuned model trained and saved.")


else:

    # ============================================================
    # TESTING / INFERENCE PHASE
    # ============================================================

    # Step 1: Load the saved model and preprocessing pipeline
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)


    # Step 2: Load the test data
    # test.csv contains both features and the actual label
    test_data = pd.read_csv(TEST_FILE)


    # Step 3: Load input.csv
    # input.csv contains only the features, without the label
    input_data = pd.read_csv("data/input.csv")


    # Step 4: Separate the actual labels from test.csv
    actual_values = test_data["median_house_value"].copy()


    # Step 5: Transform the input data using the saved pipeline
    transformed_input = pipeline.transform(input_data)


    # Step 6: Make predictions using the trained model
    predictions = model.predict(transformed_input)


    # Step 7: Calculate Mean Absolute Error
    mae = mean_absolute_error(actual_values, predictions)
    print(f"Mean Absolute Error: {mae:.2f}")


    # Step 8: Add the actual values from test.csv to input_data
    input_data["median_house_value"] = actual_values.values


    # Step 9: Add the model's predicted values
    input_data["Predicted Median House Value"] = predictions


    # Step 10: Save the actual and predicted values together
    input_data.to_csv("output.csv", index=False)

    print("Testing complete. Results saved to output.csv")