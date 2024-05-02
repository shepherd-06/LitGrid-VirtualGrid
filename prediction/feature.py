import pandas as pd
from datetime import datetime


class FeatureEngineer:
    def __init__(self, data):
        # Optionally create a copy if modifications are not intended to affect the original DataFrame.
        self.data = data.copy()

    def add_time_features(self):
        """Add time-based features to the DataFrame."""
        self.data['year'] = self.data.index.year
        self.data['month'] = self.data.index.month
        self.data['day'] = self.data.index.day
        self.data['biweek'] = self.data.index.isocalendar().week // 2

    def add_lagged_features(self, lags=3):
        """Add lagged consumption values as new features."""
        for lag in range(1, lags + 1):
            self.data.loc[:, f'lag_{lag}'] = self.data['value'].shift(lag)

    def add_rolling_average(self, window=3):
        """Add rolling average for consumption."""
        self.data.loc[:, 'rolling_avg'] = self.data['value'].rolling(
            window=window).mean()

    def prepare_features(self):
        """Prepare all features."""
        self.add_time_features()
        self.add_lagged_features()
        self.add_rolling_average()
        return self.data

    def add_future_features(self, last_date, frequency='D'):
        """
        Generate future features from last_date up to today.

        Args:
            last_date (str or pd.Timestamp): The last date in the dataset.
            frequency (str): The frequency for generating future dates ('D' for daily).

        Returns:
            pd.DataFrame: A DataFrame with future dates and their corresponding features.
        """
        # Determine the date range from the last data point to today
        today = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))
        future_dates = pd.date_range(
            start=last_date, end=today, freq=frequency)

        # Creating a DataFrame for future dates
        future_features = pd.DataFrame(index=future_dates)
        
        future_features['year'] = future_features.index.year
        future_features['month'] = future_features.index.month
        future_features['day'] = future_features.index.day
        future_features['biweek'] = future_features.index.isocalendar(
        ).week // 2

        # Adding lagged features for future predictions
        for lag in range(1, 4):  # Assuming 3 lagged features
            future_features[f'lag_{lag}'] = self.data['value'].iloc[-lag]

        # Continuing the last known rolling average
        future_features['rolling_avg'] = self.data['rolling_avg'].iloc[-1]
        return future_features
