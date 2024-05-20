import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


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
        self.training_data.index = pd.DatetimeIndex(
            self.training_data.index).to_period('h')
        self.testing_data.index = pd.DatetimeIndex(
            self.testing_data.index).to_period('h')
        # self.training_data = pd.concat([self.training_data, self.testing_data])

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
        # Convert the last index to a timestamp
        start_date = self.training_data.index[-1].to_timestamp() if hasattr(
            self.training_data.index[-1], 'to_timestamp') else self.training_data.index[-1]

        # Generate future dates
        future_dates = pd.date_range(
            start=start_date, periods=steps+1, freq='h')[1:]

        # Forecast future values
        future_forecast = self.model_fit.forecast(steps=steps)

        # Create DataFrame with forecasted values
        return pd.DataFrame({'date': future_dates, 'value': future_forecast}).set_index('date')

    def get_last_available_date(self):
        """Get the last available date from the training dataset."""
        if self.training_data.index.dtype == 'datetime64[ns]':
            last_date = self.training_data.index.max()
        else:
            raise ValueError(
                "Index is not in datetime format. Ensure the DataFrame index is datetime.")
        return last_date
