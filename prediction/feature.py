import pandas as pd
from datetime import datetime
from sklearn.impute import SimpleImputer


class FeatureEngineer:
    def __init__(self, data):
        # Optionally create a copy if modifications are not intended to affect the original DataFrame.
        self.data = data.copy()

    def add_time_features(self):
        """Add time-based features to the DataFrame."""
        self.data['year'] = self.data.index.year
        self.data['month'] = self.data.index.month
        self.data['day'] = self.data.index.day
        self.data['hour'] = self.data.index.hour
        self.data['minute'] = self.data.index.minute

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

    def add_future_features(self, model, last_date, frequency='D', steps=10):
        future_dates = pd.date_range(
            start=last_date, periods=steps, freq=frequency)
        future_features = pd.DataFrame(index=future_dates)

        # Initialize time-based features
        future_features['year'] = future_dates.year
        future_features['month'] = future_dates.month
        future_features['day'] = future_dates.day
        future_features['hour'] = future_dates.hour
        future_features['minute'] = future_dates.minute

        # Initialize lag features based on the last available values
        last_values = self.data['value'].tail(
            3).tolist()  # Assuming 3 lag features
        future_features['lag_1'] = [last_values[-1]] + [None] * (steps - 1)
        future_features['lag_2'] = [last_values[-2]] + [None] * (steps - 1)
        future_features['lag_3'] = [last_values[-3]] + [None] * (steps - 1)

        # Initialize rolling average using the available lag values
        initial_rolling_avg = sum(last_values) / len(last_values)
        future_features['rolling_avg'] = [
            initial_rolling_avg] + [None] * (steps - 1)

        # Prediction process
        for i in range(steps):
            features = future_features.iloc[i, :].drop(
                'predicted_value', errors='ignore').values.reshape(1, -1)

            # Debugging: Check for NaN values in the features before prediction
            if pd.isna(features).any():
                print(f"NaN values found in features at step {i}:")
                print(future_features.iloc[i, :])
                raise ValueError("Input X contains NaN.")

            prediction = model.predict(features)[0]
            future_features.loc[future_dates[i],
                                'predicted_value'] = prediction

            # Update the lag features and rolling average for the next prediction
            if i + 1 < steps:
                future_features.loc[future_dates[i + 1], 'lag_1'] = prediction
                future_features.loc[future_dates[i + 1],
                                    'lag_2'] = future_features.loc[future_dates[i], 'lag_1']
                future_features.loc[future_dates[i + 1],
                                    'lag_3'] = future_features.loc[future_dates[i], 'lag_2']

                # Update rolling average based on the available lag values
                lag_values = future_features.loc[future_dates[i + 1],
                                                 ['lag_1', 'lag_2', 'lag_3']].dropna()
                future_features.loc[future_dates[i + 1],
                                    'rolling_avg'] = lag_values.mean()

        return future_features
