import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pre_processing.data_processor import _DataProcessor
from pre_processing.redis_con import RedisConnector
from pre_processing.data_transformer import DataTransformer
import seaborn as sns
from pre_processing.key_mappings import get_key_by_tags, PLOT_TITLES_BY_CATEGORY
import os
from datetime import datetime


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

    def plot_decomposition(self, data_frames, title):
        """
        Plots each component from the decomposition in separate graphs and saves them.
        Args:
            data_frames (dict of pd.DataFrame): DataFrames containing the decomposed components.
            title (str): Title for the plots to specify the data type.
        """
        sns.set(style='whitegrid', context='talk', palette='pastel')

        output_dir = 'plots/'
        os.makedirs(output_dir, exist_ok=True)
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        labels = ['Observed', 'Trend', 'Seasonal', 'Residual']

        num_data_sets = len(data_frames)
        # Generate a palette with enough colors
        palette = sns.color_palette("hsv", num_data_sets)

        for idx, label in enumerate(labels):
            plt.figure(figsize=(14, 4))
            for color_idx, (description, components) in enumerate(data_frames.items()):
                data = components[idx]
                color = palette[color_idx %
                                num_data_sets]  # Assign a unique color
                sns.lineplot(data=data, color=color, label=f"{description}")
            plt.title(f"{label} - {title}")
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Description')
            plt.xticks(rotation=45)
            plt.tight_layout()
            sns.despine()

            filename = f"{label}_{title}_{current_time}.jpg"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            plt.close()
            print(f"Saved {label} plot to {filepath}")

    def analyze_seasonality(self):
        """
        Prepares data for plotting by decomposing each dataset.
        """
        decomposed_data_frames = {}
        for description, df in self.data_frames.items():
            decomposed = self.decompose(df)
            # Collect all components for plotting
            decomposed_data_frames[description] = [
                decomposed.observed, decomposed.trend,
                decomposed.seasonal, decomposed.resid
            ]
        self.plot_decomposition(decomposed_data_frames, "All Tags")


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
        data_frames=data_frames, model='additive', resample_freq=resample_freq)
    detector.analyze_seasonality()


if __name__ == "__main__":
    main()
