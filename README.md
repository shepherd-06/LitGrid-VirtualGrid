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

**Python**: >= 3.9

To run this project, you need **Python 3.9 or higher** and several packages including `redis`, `pandas`, `matplotlib`, `seaborn`, and `statsmodels`. Install all dependencies with:

```bash
pip install -r requirements.txt
```

Run the following command in the terminal (Tested on Ubuntu and MacOS):

```bash
export PYTHONPATH=$PYTHONPATH:$(realpath ../)
```

This portion is not *optimized*. The command above needs to be run before every **new** terminal session. Otherwise terminal/python won't find the required files.

### Usage

Ensure your Redis instance is operational and accessible, then run the desired module from the analysis directory to perform specific analyses or visualizations.

### Notes

- **Enhanced Module Descriptions**: The descriptions for `trend_analysis.py` and `seasonality_detection.py` are detailed to reflect their dynamic capabilities and user-driven functionality.
- **Comprehensive Overview**: This README not only outlines what each part of the project does but also how they interact and the flexibility offered to end-users. This should be helpful for both current project contributors and new stakeholders.
