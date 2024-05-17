import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score


class ProphetModelTrainer:
    def __init__(self, training_data, testing_data):
        # Ensure the DataFrame index is datetime type with frequency
        if not pd.api.types.is_datetime64_any_dtype(training_data.index):
            raise ValueError("Training data index must be datetime type")
        if not pd.api.types.is_datetime64_any_dtype(testing_data.index):
            raise ValueError("Testing data index must be datetime type")

        self.training_data = training_data.dropna()
        self.testing_data = testing_data.dropna()
        self.model = Prophet()

    def train(self):
        df = self.training_data.reset_index().rename(
            columns={'timestamp': 'ds', 'value': 'y'})
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
