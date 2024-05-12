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

    # def add_future_features(self, model, last_date, frequency='D', steps=10):
    #     """
    #     Generate and predict future features from last_date up to the number of steps specified.

    #     Args:
    #         model (object): The trained model to use for predictions.
    #         last_date (datetime or str): The last available date in the dataset.
    #         frequency (str): The frequency string for the future data points (e.g., 'h' for hourly).
    #         steps (int): Number of future periods to predict.
    #     """
    #     future_dates = pd.date_range(start=last_date, periods=steps, freq=frequency)
    #     future_features = pd.DataFrame(index=future_dates)

    #     # Initialize time-based features
    #     future_features['year'] = future_dates.year
    #     future_features['month'] = future_dates.month
    #     future_features['day'] = future_dates.day
    #     future_features['hour'] = future_dates.hour
    #     future_features['minute'] = future_dates.minute

    #     # Initialize lag features based on the last available values
    #     last_values = self.data.tail(3)['value'].tolist()
    #     future_features['lag_1'] = [last_values[-1]] + [None] * (steps - 1)
    #     future_features['lag_2'] = [last_values[-2]] + [None] * (steps - 1)
    #     future_features['lag_3'] = [last_values[-3]] + [None] * (steps - 1)
    #     future_features['rolling_avg'] = [self.data['rolling_avg'].iloc[-1]] + [None] * (steps - 1)

    #     # Initialize placeholder for predictions but do not use in model prediction
    #     future_features['predicted_value'] = [None] * steps  # this column is only for output

    #     # Apply imputation
    #     imputer = SimpleImputer(strategy='mean', fill_value=0)
    #     imputed_data = imputer.fit_transform(future_features.drop(columns=['predicted_value'], errors='ignore'))
    #     future_features_imputed = pd.DataFrame(
    #         imputed_data,
    #         index=future_dates,
    #         columns=[col for col in future_features.columns if col != 'predicted_value']
    #     )

    #     print("--------------")
    #     print("Future Prepared")
    #     print("--------------")
    #     print(future_features.head())
    #     print(future_features.tail())
    #     print("--------------")
    #     print("--------------")
            
        

    #     # Predict future values iteratively and update lag and rolling average features
    #     for i in range(steps):
    #         # if i >= 3:  # Update lag features only after having sufficient prior predictions
    #         #     future_features.loc[future_features.index[i], 'lag_1'] = future_features.iloc[i - 1]['predicted_value']
    #         #     future_features.loc[future_features.index[i], 'lag_2'] = future_features.iloc[i - 2]['predicted_value']
    #         #     future_features.loc[future_features.index[i], 'lag_3'] = future_features.iloc[i - 3]['predicted_value']
    #         # if i > 0:  # Update rolling average based on past predictions
    #         #     window = 3  # Window size for the rolling average
    #         #     future_features.loc[future_features.index[i], 'rolling_avg'] = future_features['predicted_value'].rolling(window=window).mean().iloc[i]

    #         # Prepare features for model prediction by excluding 'predicted_value'
    #         features = future_features_imputed.iloc[i].values.reshape(1, -1)  # Ensure 'predicted_value' is not included
    #         prediction = model.predict(features)[0]
    #         future_features.loc[future_dates[i], 'predicted_value'] = prediction

    #     return future_features  # return with 'predicted_value' filled with predictions


    def add_future_features(self, model, last_date, frequency='D', steps=10):
        future_dates = pd.date_range(start=last_date, periods=steps, freq=frequency)
        future_features = pd.DataFrame(index=future_dates)

        # Initialize time-based features
        future_features['year'] = future_dates.year
        future_features['month'] = future_dates.month
        future_features['day'] = future_dates.day
        future_features['hour'] = future_dates.hour
        future_features['minute'] = future_dates.minute

        # Initialize lag features based on the last available values
        last_values = self.data.tail(3)['value'].tolist()  # Assuming 3 lag features
        future_features['lag_1'] = [last_values[-1]] + [None] * (steps - 1)
        future_features['lag_2'] = [last_values[-2]] + [None] * (steps - 1)
        future_features['lag_3'] = [last_values[-3]] + [None] * (steps - 1)
        future_features['rolling_avg'] = [self.data['rolling_avg'].iloc[-1]] + [None] * (steps - 1)

        # Use imputation to handle missing values
        imputer = SimpleImputer(strategy='mean', fill_value=0)
        imputed_data = imputer.fit_transform(future_features)
        future_features_imputed = pd.DataFrame(
            imputed_data,
            index=future_dates,
            columns=[col for col in future_features.columns if col != 'predicted_value']
        )

        # Prediction process
        for i in range(steps):
            features = future_features_imputed.iloc[i].values.reshape(1, -1)
            prediction = model.predict(features)[0]
            future_features.loc[future_dates[i], 'predicted_value'] = prediction

        return future_features
