import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


class SARIMAModelTrainer:
    def __init__(self, training_data, testing_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        # Ensure the DataFrame index is datetime type with frequency
        if not pd.api.types.is_datetime64_any_dtype(training_data.index):
            raise ValueError("Training data index must be datetime type")
        if not pd.api.types.is_datetime64_any_dtype(testing_data.index):
            raise ValueError("Testing data index must be datetime type")

        self.training_data = training_data.dropna()
        self.testing_data = testing_data.dropna()
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None

    def train(self):
        series = self.training_data['value']
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mse_scores = []
        r2_scores = []

        for train_index, val_index in kf.split(series):
            train, val = series.iloc[train_index], series.iloc[val_index]

            # Fit SARIMA model
            model = SARIMAX(train, order=self.order,
                            seasonal_order=self.seasonal_order)
            model_fit = model.fit(disp=False)
            predictions = model_fit.predict(
                start=len(train), end=len(train)+len(val)-1, dynamic=False)
            mse_scores.append(mean_squared_error(val, predictions))
            r2_scores.append(r2_score(val, predictions))

        self.model_fit = model_fit  # Save the last fit model
        print(f'Average MSE: {np.mean(mse_scores)}')
        print(f'Average R2: {np.mean(r2_scores)}')

    def evaluate(self):
        series = self.testing_data['value']
        predictions = self.model_fit.predict(start=len(self.training_data), end=len(
            self.training_data)+len(series)-1, dynamic=False)
        mse = mean_squared_error(series, predictions)
        r2 = r2_score(series, predictions)
        return mse, r2

    def predict(self, steps):
        future_dates = pd.date_range(
            start=self.training_data.index[-1], periods=steps+1, freq='H')[1:]
        future_forecast = self.model_fit.get_forecast(
            steps=steps).predicted_mean
        return pd.DataFrame({'date': future_dates, 'value': future_forecast}).set_index('date')

    def get_last_available_date(self):
        """Get the last available date from the training dataset."""
        if self.training_data.index.dtype == 'datetime64[ns]':
            last_date = self.training_data.index.max()
        else:
            raise ValueError(
                "Index is not in datetime format. Ensure the DataFrame index is datetime.")
        return last_date
