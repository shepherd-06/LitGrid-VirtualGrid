import pandas as pd

from prediction.data_handler import DataHandler
from prediction.feature import FeatureEngineer
from prediction.model_trainer import ModelTrainer

def main():
    # Redis key for actual consumption data
    redis_key = 'electricity_consumption_actual'
    data_handler = DataHandler(redis_key=redis_key)
    train_data, test_data = data_handler.segment_data(train_size=0.9)

    # Display initial data
    print("--------------")
    print("Train Data")
    print("--------------")
    print(train_data.head())
    print(train_data.tail())

    # Prepare features for training data
    feature_engineer = FeatureEngineer(train_data)
    train_data_prepared = feature_engineer.prepare_features()
    print("--------------")
    print("Train Data Prepared")
    print("--------------")
    print(train_data_prepared.head())
    print(train_data_prepared.tail())

    # Prepare features for testing data
    feature_engineer_test = FeatureEngineer(test_data)
    test_data_prepared = feature_engineer_test.prepare_features()
    print("--------------")
    print("Test Data Prepared")
    print("--------------")
    print(test_data_prepared.head())
    print(test_data_prepared.tail())

    # Train the model
    trainer = ModelTrainer(train_data_prepared, test_data_prepared)
    trainer.train()
    mse, r2 = trainer.evaluate()
    print(f'Model MSE: {mse}')
    print(f'Model R^2: {r2}')

    # Predict future data
    last_date = trainer.get_last_available_date()
    future_features = feature_engineer.add_future_features(trainer.model, last_date, frequency='h', steps=24)
    future_predictions = trainer.predict(future_features.drop(columns=['predicted_value'], errors='ignore'))

    print("--------------")
    print("Future Predictions")
    print("--------------")
    print(pd.DataFrame(future_predictions, index=future_features.index, columns=['Prediction']).head())

    # Optionally, plotting or additional analysis could go here

if __name__ == "__main__":
    main()
