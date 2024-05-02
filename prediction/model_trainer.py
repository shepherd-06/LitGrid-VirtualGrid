import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from prediction.feature import FeatureEngineer


class ModelTrainer:
    def __init__(self, training_data, testing_data):
        # Drop rows with NaN values in training data
        self.training_data = training_data.dropna()
        # Drop rows with NaN values in testing data
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

    def get_last_available_date(self):
        """Get the last available date from the training dataset."""
        last_date = self.training_data.index[-1]
        return last_date

    def predict_up_to_today(self, future_features):
        """
        Predict up to today using the trained model.

        Args:
            last_date (str or pd.Timestamp): The last date from the available dataset.
            frequency (str): The frequency of prediction ('D' for daily).

        Returns:
            pd.DataFrame: Predictions from the last known data point up to today.
        """
        predictions = self.model.predict(future_features)
        return pd.DataFrame(data=predictions, index=future_features.index, columns=['Prediction'])

    def plot_predictions(self, predictions, train_data, test_data):
        """Plot the predictions along with training and test data."""
        plt.figure(figsize=(10, 5))

        # Plot training data
        plt.plot(train_data.index,
                 train_data['value'], label='Training Data', color='blue')

        # Plot test data
        plt.plot(test_data.index,
                 test_data['value'], label='Test Data', color='green')

        # Plot predictions
        plt.plot(predictions.index,
                 predictions['Prediction'], label='Predicted', color='red')

        # Set plot labels and legend
        plt.title('Predictions Along with Training and Test Data')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
