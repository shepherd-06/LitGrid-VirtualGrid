from prediction.data_handler import DataHandler
from prediction.feature import FeatureEngineer
from prediction.model_trainer import ModelTrainer


def main():
    redis_key = 'electricity_consumption_actual'
    data_handler = DataHandler(redis_key=redis_key)
    train_data, test_data = data_handler.segment_data(train_size=0.9)

    future_engineering = FeatureEngineer(train_data)
    train_data_prepared = future_engineering.prepare_features()

    future_engineering = FeatureEngineer(test_data)
    test_data_prepared = future_engineering.prepare_features()

    trainer = ModelTrainer(train_data_prepared, test_data_prepared)
    trainer.train()
    mse, r2 = trainer.evaluate()

    print(f'Model MSE: {mse}')
    print(f'Model R^2: {r2}')

    last_date = trainer.get_last_available_date()
    future_features = future_engineering.add_future_features(last_date)
    print(last_date)
    print(future_features.describe())
    print("-------------")
    print(future_features.info())
    future_df = trainer.predict_up_to_today(future_features=future_features)
    trainer.plot_predictions(future_df, train_data, test_data)


# def main():
#     # Redis key for actual consumption data
#     redis_key = 'electricity_consumption_actual'
#     data_handler = DataHandler(redis_key=redis_key)
#     train_data, test_data = data_handler.segment_data()

#     # Initialize and prepare features
#     feature_engineer = FeatureEngineer(train_data)
#     train_data_prepared = feature_engineer.prepare_features()

#     feature_engineer = FeatureEngineer(test_data)
#     test_data_prepared = feature_engineer.prepare_features()

#     # Continue with model training and evaluation
#     print("Features prepared for training and testing datasets.")
#     print(len(train_data_prepared), len(test_data_prepared))

if __name__ == "__main__":
    main()
