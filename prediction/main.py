import pandas as pd

import json

from prediction.data_handler import DataHandler
from prediction.feature import FeatureEngineer
from prediction.model_trainer import ModelTrainer
from analysis.trend_analysis import TrendAnalysis
from pre_processing.redis_con import RedisConnector


def df_to_json(df):
    """
    Converts a DataFrame to a JSON array with the format [{time: xx, value: xx}, ...]
    Args:
        df (DataFrame): The DataFrame to convert, with a datetime index and a 'value' column.
    Returns:
        str: A JSON string representing the DataFrame.
    """
    # Ensure the DataFrame index is of datetime type
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame index must be datetime type")

    # Create a list of dictionaries from the DataFrame
    json_array = [{'time': time.strftime(
        '%Y-%m-%dT%H:%M:%S'), 'value': value} for time, value in df['value'].items()]

     # Convert the list of dictionaries to a JSON string and save it to a file
    with open("test.json", 'w') as file:
        json.dump(json_array, file, indent=4)


def analyze_trends(df, freq='15D'):
    """
    Analyzes trends based on a given frequency.
    Args:
        df (DataFrame): The DataFrame to analyze.
        freq (str): Frequency for resampling ('M' for monthly, '15D' for biweekly).
    Returns:
        DataFrame: The DataFrame with added trend analysis columns.
    """
    if freq:
        # Resample the DataFrame based on the specified frequency and calculate the maximum value
        max_resampled_df = df.resample(freq).max()
        min_resampled_df = df.resample(freq).min()
        avg_max_min_resampled_df = (max_resampled_df + min_resampled_df) / 2

    # Calculate the rolling average with the average of the maximum and minimum values
    avg_max_min_resampled_df['rolling_avg'] = avg_max_min_resampled_df['value'].rolling(
        window=3, min_periods=1).mean()

    # Combine the results into a single DataFrame
    # result_df = max_resampled_df.copy()
    # result_df['rolling_avg_max_min'] = avg_max_min_resampled_df['rolling_avg_max_min']

    return avg_max_min_resampled_df


def main():
    redis_key = 'electricity_consumption_actual'
    data_handler = DataHandler(redis_key=redis_key)
    train_data, test_data = data_handler.segment_data()

    # Initialize the FeatureEngineer and prepare features
    feature_engineer = FeatureEngineer(train_data)
    train_features = feature_engineer.prepare_features()

    feature_engineer_test = FeatureEngineer(test_data)
    test_features = feature_engineer_test.prepare_features()

    # Initialize ModelTrainer
    trainer = ModelTrainer(train_features, test_features)

    # Train the model
    trainer.train()
    trainer.evaluate()

    # Predict future data
    last_date = trainer.get_last_available_date()
    future_features = feature_engineer.add_future_features(
        trainer.model, last_date, frequency='h', steps=12000)

    # Check for NaN values and handle them
    if future_features.isna().any().any():
        future_features = future_features.fillna(
            0)  # or any other method to handle NaNs

    # Predict future values
    future_predictions = trainer.predict(future_features.drop(
        columns=['predicted_value'], errors='ignore'))
    future_predictions_df = pd.DataFrame(
        future_predictions, index=future_features.index, columns=['value'])

    print("--------------")
    print("Future Predictions")
    print("--------------")
    print(future_predictions_df.head())
    print(future_predictions_df.tail())
    
    print("--------------------------")
    df_to_json(future_predictions_df)

    # # Analyze trends on the future predictions
    # trend_analysis = TrendAnalysis(RedisConnector().get_connection())
    # analyzed_df = analyze_trends(future_predictions_df)
    # # print(future_predictions_df.columns)

    # print("--------------")
    # print("Resampled Stuff")
    # print("--------------")
    # print(analyzed_df.head())
    # print(analyzed_df.tail())
    # # Plot the results
    # data_frames = {"Future Predictions": analyzed_df}
    # trend_analysis.plot_trends(
    #     data_frames, "Future Predictions Trends", "Value (MW)")


if __name__ == "__main__":
    main()
