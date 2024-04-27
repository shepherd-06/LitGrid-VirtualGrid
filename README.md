# ECIU Challenge - Virtual Grid

This project is focused on analyzing the trends in electricity consumption in Lithuania by processing and analyzing time-series data. The goal is to identify patterns, seasonality, and outliers in electricity generation and consumption, offering insights for optimizing electricity storage and distribution.

## Project Structure

The project consists of several Python modules, each performing specific roles in the data processing and analysis pipeline:

### pre_processing/

- `data_processor.py`: Handles the retrieval of electricity data files and stores them in Redis.
- `data_transformer.py`: Transforms raw data from Redis into a pandas DataFrame suitable for analysis.
- `json_loader.py`: Loads JSON data into the system before processing.
- `key_mappings.py`: Maintains a mapping between data file descriptions and Redis key formats.
- `redis_con.py`: Manages Redis connection parameters and interactions.

### analysis/

- `consumption_trend_analysis.py`: Conducts comprehensive trend analysis on electricity consumption data, including data fetching, statistical methods application, and visualization of trends.
- `seasonality_detection.py`: Examines time-series data to extract and analyze seasonal patterns in electricity consumption using seasonal decomposition.
- `outlier_detection.py`: Implements algorithms to detect anomalies in the data, indicating unusual consumption patterns or data issues.
- `distribution_visualizer.py`: Visualizes the frequency distribution of electricity consumption values, helping to identify skewness or bimodality.
- `key_metric_calculator.py`: Calculates key metrics such as mean, median, standard deviation, min, and max values to understand the distributions and variability in your data.

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

## Requirements

To run this project, you need Python 3.x and several packages including `redis`, `pandas`, `matplotlib`, `seaborn`, and `statsmodels`. Install all dependencies with:

```bash
pip install -r requirements.txt
```

### Notes

1. **Modules Described**: Each module's purpose is briefly described to give a clear understanding of its role in the project.
2. **Dynamic Nature of Scripts**: Descriptions now reflect the flexibility and dynamic capabilities of the scripts like `distribution_visualizer.py` and `outlier_detection.py`.
3. **Project End State**: Updated to reflect new capabilities added to the project.

This README provides a comprehensive overview of the current state of the project and its capabilities, designed to keep project contributors and stakeholders well-informed.
