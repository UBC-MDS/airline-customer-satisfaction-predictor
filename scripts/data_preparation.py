# data_preparation.py
# author: Hrayr Muradyan
# date: 2024-12-02

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from pathlib import Path
import pickle
from data_validation import validate_data

def clean_column_names(df):
    """
    Clean column names by converting them to lowercase, replacing spaces and dashes with underscores. 
    
    Parameters:
    df (pd.DataFrame): The input pandas DataFrame whose column names need to be cleaned.

    Returns:
    pd.DataFrame: A DataFrame with cleaned column names.
    """
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(r'\s+', '_', regex=True)
    df.columns = df.columns.str.replace(r'-', '_', regex=True)
    df = df.rename({"departure/arrival_time_convenient": "time_convenient"}, axis=1)

    return df

def correct_precision_after_scaling(df, ordinal_features, decimals=2):
    df[ordinal_features] = df[ordinal_features].astype("float32")

@click.command()
@click.option('--raw_data', type=str, help="Path to a raw data")
@click.option('--test_size', type=float, help="The proportion of the data points allocated to the test set", default=0.2)
@click.option('--data_to', type=str, help="Path to directory where the processed dataset will be saved")
@click.option('--preprocessor_to', type=str, help="Path to the directory where the preprocessor file will be saved")
@click.option('--seed', type=int, help= "Random seed", default=42)
def main(raw_data, test_size, data_to, preprocessor_to, seed):
    np.random.seed(seed)
    raw_data = pd.read_csv(raw_data)
    
    data_to = Path(data_to)
    raw_data_directory = data_to / "raw"
    processed_data_directory = data_to / "processed"

    preprocessor_to = Path(preprocessor_to)

    # If the directories do not exist, create them
    if not raw_data_directory.exists():
        raw_data_directory.mkdir(parents=True, exist_ok=True)

    if not processed_data_directory.exists():
        processed_data_directory.mkdir(parents=True, exist_ok=True)

    if not preprocessor_to.exists():
        preprocessor_to.mkdir(parents=True, exist_ok=True)

    # Clean the column names
    satisfaction_data = clean_column_names(raw_data)

    # Customer Type column's values were not homogeneous, renamed disloyal Customer -> Disloyal Customer
    satisfaction_data["customer_type"] = satisfaction_data["customer_type"].str.title()

    # Validate data
    validate_data(satisfaction_data, missing_data_threshold=0.05)

    # Train-Test Split
    train_data, test_data = train_test_split(
        satisfaction_data, test_size=test_size, random_state=seed
    )

    raw_data_directory = data_to / "raw"

    if not raw_data_directory.exists():
        raw_data_directory.mkdir(parents=True, exist_ok=True)

    # Save the splitted datasets
    train_data.to_csv(raw_data_directory / "satisfaction_train.csv", index=False)
    test_data.to_csv(raw_data_directory / "satisfaction_test.csv", index=False)

    # Print about saving the raw data in the terminal
    print(f"Raw data is saved in the directory: {raw_data_directory}")

    # Define column types
    categorical_cols = ['gender', 'customer_type', 'type_of_travel', 'class']

    ordinal_cols = ['inflight_wifi_service', 'time_convenient', 'ease_of_online_booking', 
                'gate_location', 'food_and_drink', 'online_boarding', 'seat_comfort', 
                'inflight_entertainment', 'on_board_service', 'leg_room_service',
               'baggage_handling', 'checkin_service', 'inflight_service', 'cleanliness']
    
    numerical_cols = ['age', 'flight_distance', 'departure_delay_in_minutes']
    
    drop_cols = ['arrival_delay_in_minutes', 'id']

    passthrough_cols = ['satisfaction']
    assert len(categorical_cols + ordinal_cols + numerical_cols + drop_cols + passthrough_cols) == len(train_data.columns), \
    "The sum of the number of columns in the categorical_cols, ordinal_cols and numerical cols is not equal to the number of columns in train data"

    # Define the preprocessor
    preprocessor = make_column_transformer(
        (OneHotEncoder(drop='first', handle_unknown='ignore', dtype=np.int32), categorical_cols),
        (MinMaxScaler(), ordinal_cols),
        (StandardScaler(), numerical_cols),
        ('drop', drop_cols),
        remainder='passthrough'
    )

    # Save the preprocessor
    pickle.dump(preprocessor, open(preprocessor_to / "preprocessor.pickle", "wb"))

    # Fit the preprocessor
    preprocessor.fit(train_data)

    # Preprocess the train and test sets
    scaled_train_array = preprocessor.transform(train_data)
    scaled_test_array = preprocessor.transform(test_data)

    # Get the columns
    ohe_cols = preprocessor.transformers_[0][1].get_feature_names_out(categorical_cols)
    all_cols = list(ohe_cols) + ordinal_cols + numerical_cols + passthrough_cols

    # Convert the numpy array to dataframe
    scaled_train_df = pd.DataFrame(scaled_train_array, columns=all_cols)
    scaled_test_df = pd.DataFrame(scaled_test_array, columns=all_cols)

    # Some ordinal features have precision error, this function fixes it
    correct_precision_after_scaling(scaled_train_df, ordinal_cols)
    correct_precision_after_scaling(scaled_test_df, ordinal_cols)

    # Save the scaled data
    scaled_train_df.to_csv(processed_data_directory / "scaled_satisfaction_train.csv", index=False)
    scaled_test_df.to_csv(processed_data_directory / "scaled_satisfaction_test.csv", index=False)

    # Print about saving the scaled data in the terminal
    print(f"Processed data is saved in the directory: {processed_data_directory}")

if __name__ == '__main__':
    main()