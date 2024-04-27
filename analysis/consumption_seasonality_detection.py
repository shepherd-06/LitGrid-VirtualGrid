import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pre_processing.data_processor import _DataProcessor
from pre_processing.redis_con import RedisConnector
from pre_processing.data_transformer import DataTransformer
import seaborn as sns
from pre_processing.key_mappings import get_key_by_tags, PLOT_TITLES_BY_CATEGORY


class SeasonalityDetector:
    def __init__(self, data_frames, model='additive', resample_freq='ME'):
        """
        Initializes the SeasonalityDetector with multiple data frames.
        Args:
            data_frames (dict of pd.DataFrame): Dictionary containing actual, planned, and projected data frames.
            model (str): Decomposition model ('additive' or 'multiplicative').
            resample_freq (str): The frequency for resampling data ('M' for monthly, '15D' for biweekly).
        """
        self.data_frames = data_frames
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
            title (str): Title for the plots to specify the data type.
        """
        sns.set(style='whitegrid', context='talk', palette='pastel')
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        sns.lineplot(data=result.observed, ax=axes[0], color='blue').set_title(
            f"Observed - {title}")
        sns.lineplot(data=result.trend, ax=axes[1], color='green').set_title(
            f"Trend - {title}")
        sns.lineplot(data=result.seasonal, ax=axes[2], color='orange').set_title(
            f"Seasonal - {title}")
        sns.lineplot(data=result.resid, ax=axes[3], color='red').set_title(
            f"Residual - {title}")
        plt.tight_layout()
        sns.despine()
        plt.show()

    def analyze_seasonality(self):
        """
        Performs seasonal decomposition on resampled data for each dataset and plots the results.
        """
        for label, df in self.data_frames.items():
            result = self.decompose(df)
            self.plot_decomposition(result, label)


def main():
    redis_connector = RedisConnector().get_connection()
    data_processor = _DataProcessor(redis_connector=redis_connector)

    tag = input("Enter category (consumption, production, source, storage): ")
    tags = get_key_by_tags(tag)
    data_frames = {}

    for description, key in tags.items():
        print(f"Fetching and processing data for {description}")
        raw_data = data_processor.fetch_data_by_keytype(key)
        transformer = DataTransformer(raw_data)
        df = transformer.transform_to_dataframe()
        data_frames[description] = df

    # Initialize and analyze seasonality
    resample_freq = input(
        "Enter resampling frequency ('ME' for monthly, '15D' for biweekly): ")
    detector = SeasonalityDetector(
        data_frames, model='additive', resample_freq=resample_freq)
    detector.analyze_seasonality()


if __name__ == "__main__":
    main()
