import pandas as pd

from prediction.data_handler import DataHandler
from prediction.feature import FeatureEngineer
from prediction.model_trainer import ModelTrainer
from analysis.trend_analysis import TrendAnalysis
from pre_processing.redis_con import RedisConnector

def main():
    # Redis key for actual consumption data
    redis_key = 'electricity_consumption_actual'
    data_handler = DataHandler(redis_key=redis_key)
    train_data, test_data = data_handler.segment_data(train_size=0.9)

    # Display initial data
    # print("--------------")
    # print("Train Data")
    # print("--------------")
    # print(train_data.head())
    # print(train_data.tail())

    # Prepare features for training data
    feature_engineer = FeatureEngineer(train_data)
    train_data_prepared = feature_engineer.prepare_features()
    # print("--------------")
    # print("Train Data Prepared")
    # print("--------------")
    # print(train_data_prepared.head())
    # print(train_data_prepared.tail())

    # Prepare features for testing data
    feature_engineer_test = FeatureEngineer(test_data)
    test_data_prepared = feature_engineer_test.prepare_features()
    # print("--------------")
    # print("Test Data Prepared")
    # print("--------------")
    # print(test_data_prepared.head())
    # print(test_data_prepared.tail())

    # Train the model
    trainer = ModelTrainer(train_data_prepared, test_data_prepared)
    trainer.train()
    mse, r2 = trainer.evaluate()
    # print(f'Model MSE: {mse}')
    # print(f'Model R^2: {r2}')

    # Predict future data
    last_date = trainer.get_last_available_date()
    future_features = feature_engineer.add_future_features(trainer.model, last_date, frequency='h', steps=9600)
    
    # Check where NaN values are
    nan_summary = future_features.isna().sum()
    print("NaN values per column:\n", nan_summary)

    # Display rows with NaNs
    rows_with_nans = future_features[future_features.isna().any(axis=1)]
    print("Rows with NaN values:\n", rows_with_nans)

    # Optional: Fill NaN values with a specific value (e.g., 0 or mean)
    # future_features = future_features.fillna(0)  # Example of filling NaNs with 0

    # Optional: Remove rows with NaNs
    # future_features = future_features.dropna()

    # Ensure no NaNs before prediction
    if future_features.isna().any().any():
        raise ValueError("There are still NaN values in future_features. Handle them before prediction.")

    future_predictions = trainer.predict(future_features.drop(columns=['predicted_value'], errors='ignore'))

     # Create a DataFrame for the future predictions
    future_predictions_df = pd.DataFrame(future_predictions, index=future_features.index, columns=['value'])

    print("--------------")
    print("Future Predictions")
    print("--------------")
    print(future_predictions_df.head())
    print(future_predictions_df.tail())
    # Optionally, plotting or additional analysis could go here
    # Analyze trends on the future predictions
    trend_analysis = TrendAnalysis(RedisConnector().get_connection())
    analyzed_df = trend_analysis.analyze_trends(future_predictions_df, '15D')
     # Plot the results
    data_frames = {"Future Predictions": analyzed_df}
    trend_analysis.plot_trends(data_frames, "Future Predictions Trends", "Value (MW)")



if __name__ == "__main__":
    main()
