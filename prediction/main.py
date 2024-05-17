import pandas as pd

import json

from prediction.data_handler import DataHandler
from prediction.feature import FeatureEngineer
from prediction.model_trainer import ModelTrainer, ARIMAModelTrainer
from analysis.trend_analysis import TrendAnalysis
from pre_processing.redis_con import RedisConnector


def df_to_json(df, filename):
    """
    Converts a DataFrame to a JSON array with the format [{time: xx, value: xx}, ...]
    Args:
        df (DataFrame): The DataFrame to convert, with a datetime index and a 'value' column.
    Returns:
        str: A JSON string representing the DataFrame.
    """
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame index must be datetime type")

    json_array = [{'time': time.strftime(
        '%Y-%m-%dT%H:%M:%S'), 'value': value} for time, value in df['value'].items()]

    with open(f"output/test_{filename}.json", 'w') as file:
        json.dump(json_array, file, indent=4)


def analyze_trends(df, freq='15D'):
    if freq:
        max_resampled_df = df.resample(freq).max()
        min_resampled_df = df.resample(freq).min()
        avg_max_min_resampled_df = (max_resampled_df + min_resampled_df) / 2

    avg_max_min_resampled_df['rolling_avg'] = avg_max_min_resampled_df['value'].rolling(
        window=3, min_periods=1).mean()
    return avg_max_min_resampled_df


def run_linear_regression(train_data, test_data):
    feature_engineer = FeatureEngineer(train_data)
    train_features = feature_engineer.prepare_features()

    feature_engineer_test = FeatureEngineer(test_data)
    test_features = feature_engineer_test.prepare_features()

    trainer = ModelTrainer(train_features, test_features)
    trainer.train()
    mse, r2 = trainer.evaluate()
    print(f'Linear Regression - MSE: {mse}, R2: {r2}')

    last_date = trainer.get_last_available_date()
    future_features = feature_engineer.add_future_features(
        trainer.model, last_date, frequency='H', steps=2400)

    if future_features.isna().any().any():
        future_features = future_features.fillna(0)

    future_predictions = trainer.predict(future_features.drop(
        columns=['predicted_value'], errors='ignore'))
    future_predictions_df = pd.DataFrame(
        future_predictions, index=future_features.index, columns=['value'])

    print("--------------")
    print("Future Predictions")
    print("--------------")
    print(future_predictions_df.head())
    print(future_predictions_df.tail())
    df_to_json(future_predictions_df, "linear")


def run_arima(train_data, test_data):
    arima_trainer = ARIMAModelTrainer(train_data, test_data, order=(5, 1, 0))
    arima_trainer.train()
    mse, r2 = arima_trainer.evaluate()
    print(f'ARIMA - MSE: {mse}, R2: {r2}')

    future_predictions = arima_trainer.predict(steps=2400)
    print(f'Future Predictions:\n{future_predictions}')

    df_to_json(future_predictions, "arima")


def main():
    redis_key = 'electricity_consumption_actual'
    data_handler = DataHandler(redis_key=redis_key)
    train_data, test_data = data_handler.segment_data()

    model_choice = input(
        "Choose the model you want to work with (linear_regression[1]/arima[2]): ").strip().lower()

    if model_choice == '1':
        run_linear_regression(train_data, test_data)
    elif model_choice == '2':
        run_arima(train_data, test_data)
    else:
        print("Invalid choice. Please choose either linear_regression => [1] or arima => [2].")


if __name__ == "__main__":
    main()
