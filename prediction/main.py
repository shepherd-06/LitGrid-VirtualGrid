import os
import pandas as pd

import json
import warnings
import seaborn as sns
import matplotlib.pyplot as plt


from prediction.data_handler import DataHandler
from prediction.feature import FeatureEngineer
from prediction.model_trainer import ModelTrainer, ARIMAModelTrainer
from prediction.prophet_model_trainer import ProphetModelTrainer
from prediction.sarima_model_trainer import SARIMAModelTrainer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Suppress specific warning
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels.tsa.base.tsa_model",
                        message="No supported index is available. Prediction results will be given with an integer index beginning at `start`.")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels.tsa.base.tsa_model",
                        message="No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.")


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


def run_linear_regression(train_data, test_data, steps=7200):
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
        trainer.model, last_date, frequency='h', steps=steps)

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


def run_arima(train_data, test_data, steps=7200):
    arima_trainer = ARIMAModelTrainer(train_data, test_data, order=(5, 1, 0))
    arima_trainer.train()
    mse, r2 = arima_trainer.evaluate()
    print(f'ARIMA - MSE: {mse}, R2: {r2}')

    future_predictions = arima_trainer.predict(steps=steps)
    print(f'Future Predictions:\n{future_predictions}')

    df_to_json(future_predictions, "arima")


def run_prophet(train_data, test_data, steps=7200):
    prophet_trainer = ProphetModelTrainer(train_data, test_data)
    prophet_trainer.train()
    mse, r2 = prophet_trainer.evaluate()
    print(f'Prophet - MSE: {mse}, R2: {r2}')

    future_predictions = prophet_trainer.predict(steps=steps)
    print(f'Future Predictions:\n{future_predictions}')

    df_to_json(future_predictions, "prophet")


def run_sarima(train_data, test_data, steps=7200):
    sarima_trainer = SARIMAModelTrainer(
        train_data, test_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
    sarima_trainer.train()
    mse, r2 = sarima_trainer.evaluate()
    print(f'SARIMA - MSE: {mse}, R2: {r2}')

    future_predictions = sarima_trainer.predict(steps=steps)
    print(f'Future Predictions:\n{future_predictions}')

    df_to_json(future_predictions, "sarima")


def plot_predictions(test_data, output_folder='output'):
    model_files = {
        'ARIMA': 'test_arima.json',
        'Linear Regression': 'test_linear.json',
        'Prophet': 'test_prophet.json',
        'SARIMA': 'test_sarima.json'
    }

    test_data = test_data.resample('D').mean().reset_index()
    test_data['model'] = "Test Sample"

    all_predictions = [test_data]

    for model_name, file_name in model_files.items():
        file_path = os.path.join(output_folder, file_name)

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        with open(file_path, 'r') as f:
            predictions_json = json.load(f)

        predictions_df = pd.DataFrame(predictions_json)
        predictions_df['time'] = pd.to_datetime(predictions_df['time'])
        predictions_df.set_index('time', inplace=True)

        predictions_df = predictions_df.resample('D').mean().reset_index()
        predictions_df['model'] = model_name

        all_predictions.append(predictions_df)

    combined_df = pd.concat(all_predictions)

    plt.figure(figsize=(14, 7))
    sns.lineplot(x='timestamp', y='value', hue='model', data=combined_df)
    plt.title('Model Predictions vs Test Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(title='Model')
    plt.grid(True)
    plt.show()


def plot_predictions_v2(model_name):
    model_files = {
        'ARIMA': 'test_arima.json',
        'Linear Regression': 'test_linear.json',
        'Prophet': 'test_prophet.json',
        'SARIMA': 'test_sarima.json'
    }
    
    file_name = model_files[model_name]
    file_path = os.path.join("output", file_name)
    
    if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
        
    with open(file_path, 'r') as f:
            predictions_json = json.load(f)

    predictions_df = pd.DataFrame(predictions_json)
    predictions_df['time'] = pd.to_datetime(predictions_df['time'])
    predictions_df.set_index('time', inplace=True)

    predictions_df = predictions_df.resample('D').mean().reset_index()
    predictions_df['model'] = model_name

    plt.figure(figsize=(14, 7))
    sns.lineplot(x='time', y='value', hue='model', data=predictions_df)
    plt.title('Model Predictions vs Test Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(title='Model')
    plt.grid(True)
    plt.show()


def evaluate_predictions(test_data, output_folder='output'):
    model_files = {
        'ARIMA': 'test_arima.json',
        'Linear Regression': 'test_linear.json',
        'Prophet': 'test_prophet.json',
        'SARIMA': 'test_sarima.json'
    }

    results = []
    print("test_data")
    print(test_data.head())

    for model_name, file_name in model_files.items():
        file_path = os.path.join(output_folder, file_name)

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        with open(file_path, 'r') as f:
            predictions_json = json.load(f)

        predictions_df = pd.DataFrame(predictions_json)
        predictions_df['time'] = pd.to_datetime(predictions_df['time'])
        predictions_df.set_index('time', inplace=True)
        print('-------------')
        print(model_name)
        print(predictions_df.head())

        # Ensure we are only comparing the times that exist in both datasets
        common_index = test_data.index.intersection(predictions_df.index)

        if common_index.empty:
            print(f"No common timestamps found for model: {model_name}")
            continue

        y_true = test_data.loc[common_index, 'value']
        y_pred = predictions_df.loc[common_index, 'value']

        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        results.append({
            'model': model_name,
            'mse': mse,
            'r2': r2,
            'mae': mae
        })

    plot_predictions(test_data)
    return results


def main():
    redis_key = 'electricity_consumption_actual'
    data_handler = DataHandler(redis_key=redis_key)
    train_data, test_data = data_handler.segment_data()

    model_choice = input(
        "Choose the model you want to work with (linear_regression[1]/arima[2]/prophet[3]/sarima[4]/test[5]/specific graph[6]): ").strip().lower()

    if model_choice == '1' or model_choice == 'linear_regression':
        run_linear_regression(train_data, test_data)
    elif model_choice == '2' or model_choice == 'arima':
        run_arima(train_data, test_data)
    elif model_choice == '3' or model_choice == 'prophet':
        run_prophet(train_data, test_data)
    elif model_choice == '4' or model_choice == 'sarima':
        run_sarima(train_data, test_data)
    elif model_choice == '5' or model_choice == 'test':
        results = evaluate_predictions(test_data, output_folder='output')
        for result in results:
            print(f"Model: {result['model']}")
            print(f"  MSE: {result['mse']}")
            print(f"  R²: {result['r2']}")
            print(f"  MAE: {result['mae']}")
            print()
    elif model_choice == '6':
        graph_choice = input(
            "Which graph to show: [linear[1], arima[2], prophet[3], sarima[4]]? = ")

        if graph_choice == "1":
            plot_predictions_v2("Linear Regression")
        elif graph_choice == "2":
            plot_predictions_v2("ARIMA")
        elif graph_choice == "3":
            plot_predictions_v2("Prophet")
        elif graph_choice == "4":
            plot_predictions_v2("SARIMA")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
