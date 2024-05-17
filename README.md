# ECIU Challenge - Virtual Grid

This project aims to analyze trends in electricity consumption in Lithuania by processing and analyzing time-series data. The goal is to identify patterns, seasonality, and outliers in electricity generation and consumption, offering insights for optimizing electricity storage and distribution.

## Project Structure

The project consists of several Python modules, each performing specific roles in the data processing and analysis pipeline:

### pre_processing/

- `data_processor.py`: Handles the retrieval of electricity data files and stores them in Redis.
- `data_transformer.py`: Transforms raw data from Redis into a pandas DataFrame suitable for analysis.
- `json_loader.py`: Loads JSON data into the system before processing.
- `key_mappings.py`: Maintains a mapping between data file descriptions and Redis key formats.
- `redis_con.py`: Manages Redis connection parameters and interactions.

### analysis/

- `trend_analysis.py`: Conducts comprehensive trend analysis on electricity consumption/production/source/storage data, including data fetching, statistical methods application, and visualization of trends. It supports dynamic analysis based on user-defined intervals such as biweekly and monthly trends, as well as specific month analysis with a range.
- `seasonality_detection.py`: Examines time-series data to extract and analyze seasonal patterns in electricity consumption/production/source/storage using seasonal decomposition. This module dynamically handles multiple data sets based on user-selected tags, enabling customized decomposition and visualization of seasonal, trend, and residual components.
- `outlier_detection.py`: Implements algorithms to detect anomalies in the data, indicating unusual consumption patterns or data issues.
- `distribution_visualizer.py`: Visualizes the frequency distribution of electricity consumption values, helping to identify skewness or bimodality.
- `key_metric_calculator.py`: Calculates key metrics such as mean, median, standard deviation, min, and max values to understand the distributions and variability in your data.

### prediction/

- `feature.py`: Defines the `FeatureEngineer` class for preparing features for model training, including time-based, seasonal, and lagged features. Also includes `ARIMAModelTrainer` class for training and evaluating an ARIMA model.
- `model_trainer.py`: Defines the `ModelTrainer` class for training and evaluating a linear regression model.
- `arima_model_trainer.py`: (to be added later) Defines the `ARIMAModelTrainer` class for training and evaluating an ARIMA model.
- `prophet_model_trainer.py`: Defines the `ProphetModelTrainer` class for training and evaluating a Prophet model.
- `sarima_model_trainer.py`: Defines the `SARIMAModelTrainer` class for training and evaluating a SARIMA model.
- `lstm_model_trainer.py` (to be added later): Defines the `LSTMModelTrainer` class for training and evaluating an LSTM model.
- `data_handler.py`: Handles loading and segmenting data for training and testing.

### tests/

- `test_data_processor.py`: Validates that the `_DataProcessor` class correctly fetches and processes data.
- `test_data_transformer.py`: Checks the proper transformation of raw data into a DataFrame by the `DataTransformer` class.
- `test_json_loader.py`, `test_redis_con.py`: Test JSON loading and Redis connection functionality.

### dataset/

- Stores electricity consumption and production data in JSON format.

### .github/workflows/

- `CI.yml`: Sets up continuous integration workflows for automated testing.

## Project End State (Tentative)

The envisaged end-product is a thorough analysis system that will:

- Fetch electricity consumption data automatically from a Redis database.
- Process and convert this data into a structured format amenable to time-series analysis.
- Apply statistical methods like rolling averages and exponential smoothing to reveal consumption trends.
- Use seasonal decomposition to detect and analyze seasonal patterns and outliers.
- Visualize the trends and anomalies in a clear, accessible way, potentially through a web dashboard.
- Offer insights for electricity grid management, such as load forecasting and optimizing the use of storage devices.
- Evaluate different time series forecasting models (Linear Regression, ARIMA, Prophet, SARIMA) and provide metrics to understand their performance.

## Requirements & How to Run

**Python**: >= 3.9

To run this project, you need **Python 3.9 or higher** and several packages including `redis`, `pandas`, `matplotlib`, `seaborn`, `statsmodels`, and `prophet`. Install all dependencies with:

```bash
pip install -r requirements.txt
```

Run the following command in the terminal (Tested on Ubuntu and MacOS):

```bash
export PYTHONPATH=$PYTHONPATH:$(realpath ../)
```

This portion is not optimized. The command above needs to be run before every new terminal session. Otherwise terminal/python won't find the required files.

## Usage

Ensure your Redis instance is operational and accessible, then run the desired module from the analysis directory to perform specific analyses or visualizations.

## Model Evaluation

To evaluate the predictions from different models against the test data, use the `evaluate_predictions` function. This function reads JSON files with predicted values, compares them with the actual values in the test data, and calculates metrics like MSE and R².

To run the evaluation, execute the main function with option 5. This will provide an overview of the test data and the predictions from each model.

### Example Usage

Run the following command to start the main script:

```bash
python prediction/main.py
```

When prompted, choose option 5 to evaluate the predictions:

```plaintext
Choose the model you want to work with (linear_regression[1]/arima[2]/prophet[3]/sarima[4]/evaluate[5]): 5
```

### Expected Output

The output will display the test data and predictions from each model along with the evaluation metrics:

```bash
test_data
                        value
timestamp                    
2024-01-19 06:00:00  1657.473
2024-01-19 07:00:00  1851.665
2024-01-19 08:00:00  1973.359
2024-01-19 09:00:00  1997.864
2024-01-19 10:00:00  1977.297
-------------
ARIMA
                           value
time                            
2024-01-19 06:00:00  1480.323274
2024-01-19 07:00:00  1519.524282
2024-01-19 08:00:00  1537.627514
2024-01-19 09:00:00  1538.615125
2024-01-19 10:00:00  1523.784349
-------------
Linear Regression
                        value
time                         
2024-01-19 04:00:00  1334.554
2024-01-19 05:00:00  1321.483
2024-01-19 06:00:00  1333.782
2024-01-19 07:00:00  1334.554
2024-01-19 08:00:00  1321.483
-------------
Prophet
                           value
time                            
2024-01-19 06:00:00  1595.573316
2024-01-19 07:00:00  1733.227103
2024-01-19 08:00:00  1841.305416
2024-01-19 09:00:00  1891.154692
2024-01-19 10:00:00  1889.536813
-------------
SARIMA
                           value
time                            
2024-01-19 06:00:00  1472.331402
2024-01-19 07:00:00  1493.997744
2024-01-19 08:00:00  1513.567900
2024-01-19 09:00:00  1523.902440
2024-01-19 10:00:00  1518.855126

Model: ARIMA
  MSE: 61926.20367924915
  R²: -0.07079532297430613

Model: Linear Regression
  MSE: 67092.62933332763
  R²: -0.16013043635435475

Model: Prophet
  MSE: 173319.33934561186
  R²: -1.9969467999936032

Model: SARIMA
  MSE: 63015.84206305016
  R²: -0.07695251208429355
```

### Disclaimer

The output shown above represents the first stage results of the model predictions. The MSE (Mean Squared Error) and R² (R-squared) values are currently **off the chart**, indicating that the models' **predictions are not yet accurate**. These values are expected to improve as the models are refined and reworked. Future outputs may vary significantly as the models are upgraded or enhanced based on ongoing development and testing. Please note that these initial results are part of the iterative process of model improvement and should not be taken as final.

## Notes

- Enhanced Module Descriptions: The descriptions for trend_analysis.py and seasonality_detection.py are detailed to reflect their dynamic capabilities and user-driven functionality.

- Comprehensive Overview: This README not only outlines what each part of the project does but also how they interact and the flexibility offered to end-users. This should be helpful for both current project contributors and new stakeholders.

- This updated `README.md` provides a comprehensive overview of the project, including the newly integrated time series forecasting models and the evaluation function to compare their performance.
