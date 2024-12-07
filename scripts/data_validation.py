# data_validation.py
# author: Hrayr Muradyan
# date: 2024-12-02

import pandera as pa

def check_duplicates(df):
    return not bool(df.drop('id', axis=1).duplicated().sum())

def validate_data(df, missing_data_threshold):

    schema = pa.DataFrameSchema(
        {
            "gender": pa.Column(str, pa.Check.isin(["Male", "Female"]), nullable=False),
            "customer_type": pa.Column(str, pa.Check.isin(["Loyal Customer", "Disloyal Customer"]), nullable=False),
            "age": pa.Column(int, pa.Check.between(0, 100), nullable=False),
            "type_of_travel": pa.Column(str, pa.Check.isin(["Business travel", "Personal Travel"]), nullable=False),
            "class": pa.Column(str, pa.Check.isin(["Eco", "Eco Plus", "Business"]), nullable=False),
            "flight_distance": pa.Column(int, pa.Check.greater_than(0), nullable=False),
            "inflight_wifi_service": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "time_convenient": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "ease_of_online_booking": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "gate_location": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "food_and_drink": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "online_boarding": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "seat_comfort": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "inflight_entertainment": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "on_board_service": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "leg_room_service": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "baggage_handling": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "checkin_service": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "inflight_service": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "cleanliness": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "departure_delay_in_minutes": pa.Column(int, pa.Check.greater_than_or_equal_to(0), nullable=True),
            "arrival_delay_in_minutes": pa.Column(float, pa.Check.greater_than_or_equal_to(0), nullable=True),
            "satisfaction": pa.Column(str, pa.Check.isin(["neutral or dissatisfied", "satisfied"]), nullable=False),
        },
        checks=[
            pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found!"),
            pa.Check(lambda df: (df.isna().sum() / len(df) < missing_data_threshold).all(), error=f"Some columns have more than {missing_data_threshold*100}% missing values."),
            pa.Check(check_duplicates, error = "There are duplicates observations in the dataset!")
        ])

    try:
        schema.validate(df, lazy=True)
        print("Congratulations! Data validation passed!")
    except pa.errors.SchemaErrors as e:
        print(e.failure_cases)