import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pre_processing.data_processor import _DataProcessor
from pre_processing.redis_con import RedisConnector
from pre_processing.data_transformer import DataTransformer
import seaborn as sns


class SeasonalityDetector:
    def __init__(self, actual_df, planned_df, projected_df, model='additive', resample_freq='M'):
        """
        Initializes the SeasonalityDetector with electricity consumption data.

        Args:
            actual_df, planned_df, projected_df (pd.DataFrame): DataFrames containing the actual, planned, and projected electricity consumption data.
            model (str): Decomposition model ('additive' or 'multiplicative').
            resample_freq (str): The frequency for resampling data ('M' for monthly, '15D' for biweekly).
        """
        self.actual_df = actual_df
        self.planned_df = planned_df
        self.projected_df = projected_df
        self.model = model
        self.resample_freq = resample_freq

    def decompose(self, df):
        """
        Decomposes the resampled DataFrame into seasonal, trend, and residual components.

        Args:
            df (pd.DataFrame): The DataFrame to resample and decompose.

        Returns:
            DecomposeResult: Object with seasonal, trend, and residual components.
        """
        # Resample data
        resampled_df = df.resample(self.resample_freq).mean()

        # Decompose the resampled data
        decomposition = seasonal_decompose(
            resampled_df['value'], model=self.model)
        return decomposition

    def plot_decomposition(self, result, title):
        """
        Plots the seasonal, trend, and residual components from the decomposition using Seaborn for enhanced aesthetics.

        Args:
            result (DecomposeResult): The decomposition result to plot.
            title (str): Title for the plots to specify the data type (actual, planned, projected).
        """
        # Set the aesthetic style of the plots
        sns.set(style='whitegrid', context='talk', palette='pastel')

        # Decomposition plot layout
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        # Plot observed data
        sns.lineplot(data=result.observed, ax=axes[0], color='blue')
        axes[0].set_title(f"Observed - {title}")

        # Plot trend component
        sns.lineplot(data=result.trend, ax=axes[1], color='green')
        axes[1].set_title(f"Trend - {title}")

        # Plot seasonal component
        sns.lineplot(data=result.seasonal, ax=axes[2], color='orange')
        axes[2].set_title(f"Seasonal - {title}")

        # Plot residual component
        sns.lineplot(data=result.resid, ax=axes[3], color='red')
        axes[3].set_title(f"Residual - {title}")

        # Despine the plots for a cleaner look
        sns.despine()

        # Adjust layout to make room for plot titles and to minimize overlap
        plt.tight_layout()
        plt.show()

    def analyze_seasonality(self):
        """
        Performs seasonal decomposition on resampled actual, planned, and projected data and plots the results.
        """
        # Decompose each data set
        results_actual = self.decompose(self.actual_df)
        results_planned = self.decompose(self.planned_df)
        results_projected = self.decompose(self.projected_df)

        # Plot the decomposed components
        self.plot_decomposition(results_actual, "Actual Consumption")
        # self.plot_decomposition(results_planned, "Planned Consumption")
        # self.plot_decomposition(results_projected, "Projected Consumption")


# Example Usage
if __name__ == "__main__":
    # Example DataFrames setup
    redis_connector = RedisConnector().get_connection()
    data_processor = _DataProcessor(redis_connector=redis_connector)

    actual_data = data_processor.fetch_data_by_keytype(
        "electricity_consumption_actual")
    transformer = DataTransformer(actual_data)
    actual_df = transformer.transform_to_dataframe()

    planned_data = data_processor.fetch_data_by_keytype(
        "electricity_consumption_planned")
    transformer = DataTransformer(planned_data)
    planned_df = transformer.transform_to_dataframe()

    projected_data = data_processor.fetch_data_by_keytype(
        "electricity_consumption_projected")
    transformer = DataTransformer(projected_data)
    projected_df = transformer.transform_to_dataframe()

    # Initialize and analyze seasonality
    detector = SeasonalityDetector(
        actual_df, planned_df, projected_df, model='additive', resample_freq="15D")
    detector.analyze_seasonality()
