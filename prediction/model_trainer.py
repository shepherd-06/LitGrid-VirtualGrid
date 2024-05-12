import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from prediction.feature import FeatureEngineer
from analysis.trend_analysis import TrendAnalysis
from pre_processing.redis_con import RedisConnector

class ModelTrainer:
    def __init__(self, training_data, testing_data):
        # Drop rows with NaN values in training and testing data
        self.training_data = training_data.dropna()
        self.testing_data = testing_data.dropna()
        self.model = LinearRegression()

    def train(self):
        """Train the linear regression model."""
        features = self.training_data.drop(columns=['value'])
        target = self.training_data['value']
        self.model.fit(features, target)

    def evaluate(self):
        """Evaluate the model on the testing dataset."""
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
            raise ValueError("Index is not in datetime format. Ensure the DataFrame index is datetime.")
        return last_date

    def predict_future(self, feature_engineer, steps=10):
        """Generate and predict future values based on the trained model and future feature requirements."""
        last_date = self.get_last_available_date()
        future_features = feature_engineer.add_future_features(self.model, last_date, steps=steps)
        return future_features

    def analyze_trends(self, df, freq='ME'):
        """
        Analyzes trends based on a given frequency.
        Args:
            df (DataFrame): The DataFrame to analyze.
            freq (str): Frequency for resampling ('M' for monthly, '15D' for biweekly).
        Returns:
            DataFrame: The DataFrame with added trend analysis columns.
        """
        # Check if the necessary columns exist
        # required_columns = {'year', 'month', 'day', 'hour', 'minute', 'rolling_avg'}
        # if not required_columns.issubset(df.columns):
        #     missing = required_columns - set(df.columns)
        #     raise ValueError(f"Missing required columns: {missing}")

        # # Create a datetime index from the date and time columns
        # df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
        # df.set_index('datetime', inplace=True)

        # Resample based on the new datetime index
        resampled_df = df.resample(freq).mean()
        resampled_df['rolling_avg'] = resampled_df['rolling_avg'].rolling(window=1).mean()
        resampled_df['exp_smooth'] = resampled_df['rolling_avg'].ewm(span=1, adjust=False).mean()
        return resampled_df

    
    def plot_predictions(self, predictions, train_data, test_data):
        trend_analysis = TrendAnalysis(RedisConnector().get_connection())
        # print(predictions.info())
        print(predictions.head())
        print(predictions.tail())
        # analyzed_ft_df = self.analyze_trends(predictions, '15D')
        # data_frames = {}
        # data_frames["Future Prediction Test"] = analyzed_ft_df
        # trend_analysis.plot_trends(data_frames, "Hello World", f"Electricity Consumption (MW)")

        # """Plot the predictions along with training and test data."""
        # plt.figure(figsize=(10, 5))

        # # Plot training data
        # plt.plot(train_data.index,
        #          train_data['value'], label='Training Data', color='blue')

        # # Plot test data
        # plt.plot(test_data.index,
        #          test_data['value'], label='Test Data', color='green')

        # # Plot predictions
        # plt.plot(predictions.index,
        #          predictions['Prediction'], label='Predicted', color='red')

        # # Set plot labels and legend
        # plt.title('Predictions Along with Training and Test Data')
        # plt.xlabel('Date')
        # plt.ylabel('Value')
        # plt.legend()
        # plt.show()
