# ECIU Challenge - Virtual Grid

This project is focused on analyzing the trends in electricity consumption in Lithuania. It includes a suite of Python scripts and modules to process and analyze time-series data, with the goal of identifying patterns, seasonality, and outliers in electricity generation and consumption. The analysis aims to provide insights that could aid in the optimization of electricity storage and distribution.

## Project Structure

The project is organized into several modules and scripts, each with a specific role:

### pre_processing/

- `data_processor.py`: Contains the `_DataProcessor` class that fetches electricity data files and stores them in Redis.
- `data_transformer.py`: Features the `DataTransformer` class which converts the raw data from Redis into a pandas DataFrame for analysis.
- `json_loader.py`: Utility script to load JSON data into the system before processing.
- `key_mappings.py`: Stores a mapping from data file descriptions to Redis key formats for consistent key generation.
- `redis_con.py`: Defines the Redis connection parameters and utilities to interact with the Redis database.

### analysis/

- `consumption_trend_analysis.py`: Implements the `ConsumptionTrendAnalysis` class to perform a complete trend analysis on electricity consumption data, including fetching data, applying statistical methods, and visualizing trends.

### tests/

- `test_data_processor.py`: Unit tests for the `_DataProcessor` class to ensure data is fetched and processed correctly.
- `test_data_transformer.py`: Unit tests for the `DataTransformer` class to verify the correct transformation of raw data into DataFrame.
- `test_json_loader.py`: Tests the JSON data loading functionality.
- `test_redis_con.py`: Ensures the Redis connection and utilities are functioning as expected.

### dataset/

Contains the electricity consumption and production data in JSON format.

### .github/workflows/

- `CI.yml`: Continuous integration workflow file for automated testing.

## Project End State (Tentative)

The final product of this project will be a comprehensive analysis system that:

- Automatically fetches electricity consumption data from a Redis database.
- Processes and transforms this data into a structured format suitable for time-series analysis.
- Applies rolling averages, exponential smoothing, and other statistical methods to identify consumption trends and patterns.
- Visualizes these trends in an easy-to-understand manner, potentially integrating with a web dashboard.
- Provides insights that can help with electricity grid management, such as optimizing storage device usage and improving load forecasting.

## Requirements

To run this project, you will need Python 3.x and the following packages:

- redis
- pandas
- matplotlib

Install all dependencies by running `pip install -r requirements.txt`.

## Usage

To perform the complete trend analysis, ensure your Redis instance is running and accessible, then execute:

```bash
python -m analysis.consumption_trend_analysis
```
