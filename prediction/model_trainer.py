import numpy as np

from sklearn.linear_model import LinearRegression

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



