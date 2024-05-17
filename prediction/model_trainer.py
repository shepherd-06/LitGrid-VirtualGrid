import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


class ModelTrainer:
    def __init__(self, training_data, testing_data):
        # Drop rows with NaN values in training and testing data
        self.training_data = training_data.dropna()
        self.testing_data = testing_data.dropna()
        self.model = LinearRegression()

    def train(self):
        features = self.training_data.drop(columns=['value'])
        target = self.training_data['value']

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mse_scores = []
        r2_scores = []

        for train_index, val_index in kf.split(features):
            X_train, X_val = features.iloc[train_index], features.iloc[val_index]
            y_train, y_val = target.iloc[train_index], target.iloc[val_index]

            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_val)

            mse_scores.append(mean_squared_error(y_val, predictions))
            r2_scores.append(r2_score(y_val, predictions))

        print(f'Average MSE: {np.mean(mse_scores)}')
        print(f'Average R2: {np.mean(r2_scores)}')

    def evaluate(self):
        features_test = self.testing_data.drop(columns=['value'])
        target_test = self.testing_data['value']
        predictions = self.model.predict(features_test)
        mse = mean_squared_error(target_test, predictions)
        r2 = r2_score(target_test, predictions)
        return mse, r2

    def predict(self, features):
        """Predict using the linear regression model."""
        return self.model.predict(features)

    def get_last_available_date(self):
        """Get the last available date from the training dataset."""
        if self.training_data.index.dtype == 'datetime64[ns]':
            last_date = self.training_data.index.max()
        else:
            raise ValueError(
                "Index is not in datetime format. Ensure the DataFrame index is datetime.")
        return last_date


class ARIMAModelTrainer:
    def __init__(self, training_data, testing_data, order=(5, 1, 0)):
        # Ensure the DataFrame index is datetime type
        if not pd.api.types.is_datetime64_any_dtype(training_data.index):
            raise ValueError("Training data index must be datetime type")
        if not pd.api.types.is_datetime64_any_dtype(testing_data.index):
            raise ValueError("Testing data index must be datetime type")

        # Drop rows with NaN values in training and testing data
        self.training_data = training_data.dropna()
        self.testing_data = testing_data.dropna()
        self.order = order
        self.model_fit = None

    def train(self):
        series = self.training_data['value']
        series = series.asfreq('h')  # Ensure frequency is set

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mse_scores = []
        r2_scores = []

        for train_index, val_index in kf.split(series):
            train, val = series.iloc[train_index], series.iloc[val_index]

            # Pass time index to ARIMA
            model = ARIMA(train, order=self.order)
            model_fit = model.fit()
            predictions = model_fit.predict(
                start=len(train), end=len(train)+len(val)-1, dynamic=False)
            mse_scores.append(mean_squared_error(val, predictions))
            r2_scores.append(r2_score(val, predictions))

        self.model_fit = model_fit  # Save the last fit model
        print(f'Average MSE: {np.mean(mse_scores)}')
        print(f'Average R2: {np.mean(r2_scores)}')

    def evaluate(self):
        series = self.testing_data['value']
        series = series.asfreq('h')  # Ensure frequency is set

        predictions = self.model_fit.predict(start=len(self.training_data), end=len(
            self.training_data)+len(series)-1, dynamic=False)
        mse = mean_squared_error(series, predictions)
        r2 = r2_score(series, predictions)
        return mse, r2

    def predict(self, steps):
        future_dates = pd.date_range(
            start=self.training_data.index[-1], periods=steps+1, freq='h')[1:]
        future_forecast = self.model_fit.forecast(steps=steps)
        return pd.DataFrame({'date': future_dates, 'value': future_forecast}).set_index('date')

    def get_last_available_date(self):
        """Get the last available date from the training dataset."""
        if self.training_data.index.dtype == 'datetime64[ns]':
            last_date = self.training_data.index.max()
        else:
            raise ValueError(
                "Index is not in datetime format. Ensure the DataFrame index is datetime.")
        return last_date
