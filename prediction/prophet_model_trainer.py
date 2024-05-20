import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import numpy as np


class ProphetModelTrainer:
    def __init__(self, training_data, testing_data, n_splits=5):
        # Ensure the DataFrame index is datetime type with frequency
        if not pd.api.types.is_datetime64_any_dtype(training_data.index):
            raise ValueError("Training data index must be datetime type")
        if not pd.api.types.is_datetime64_any_dtype(testing_data.index):
            raise ValueError("Testing data index must be datetime type")

        self.training_data = training_data.dropna()
        self.testing_data = testing_data.dropna()
        self.model = Prophet()
        self.n_splits = n_splits
        # self.training_data = pd.concat([self.training_data, self.testing_data])

    def train(self):
        df = self.training_data.reset_index().rename(
            columns={'timestamp': 'ds', 'value': 'y'})

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        mse_scores = []
        r2_scores = []

        for train_index, val_index in tscv.split(df):
            train_df, val_df = df.iloc[train_index], df.iloc[val_index]
            model = Prophet()
            model.fit(train_df)
            forecast = model.predict(val_df[['ds']])
            mse = mean_squared_error(val_df['y'], forecast['yhat'])
            r2 = r2_score(val_df['y'], forecast['yhat'])
            mse_scores.append(mse)
            r2_scores.append(r2)

        print(f'Average MSE: {np.mean(mse_scores)}')
        print(f'Average R2: {np.mean(r2_scores)}')

        # Fit the final model on the entire training data
        self.model.fit(df)

    def evaluate(self):
        df = self.testing_data.reset_index().rename(
            columns={'timestamp': 'ds', 'value': 'y'})
        forecast = self.model.predict(df[['ds']])
        mse = mean_squared_error(df['y'], forecast['yhat'])
        r2 = r2_score(df['y'], forecast['yhat'])
        return mse, r2

    def predict(self, steps):
        future_dates = pd.date_range(
            start=self.training_data.index[-1], periods=steps+1, freq='h')[1:]
        future = pd.DataFrame({'ds': future_dates})
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat']].set_index('ds').rename(columns={'yhat': 'value'})

    def get_last_available_date(self):
        """Get the last available date from the training dataset."""
        if self.training_data.index.dtype == 'datetime64[ns]':
            last_date = self.training_data.index.max()
        else:
            raise ValueError(
                "Index is not in datetime format. Ensure the DataFrame index is datetime.")
        return last_date
