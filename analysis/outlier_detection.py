import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pre_processing.redis_con import RedisConnector
from pre_processing.data_processor import _DataProcessor
from pre_processing.data_transformer import DataTransformer
from pre_processing.key_mappings import get_key_by_tags, PLOT_TITLES_BY_CATEGORY


class OutlierDetector:
    def __init__(self, data_frames, threshold=3):
        """
        Initializes the OutlierDetector with the specified data frames.
        Args:
            data_frames (dict of pd.DataFrame): Dictionary containing the data frames with descriptive names.
            threshold (float): The Z-score threshold to identify outliers.
        """
        self.data_frames = data_frames
        self.threshold = threshold

    def detect_outliers(self, column_name='value'):
        """
        Detects outliers in all specified DataFrames using the Z-score method.
        Args:
            column_name (str): The column name to detect outliers for in each DataFrame.
        Updates each DataFrame in self.data_frames with an 'outlier' column indicating outliers.
        """
        for name, df in self.data_frames.items():
            if column_name in df.columns:
                z_scores = np.abs(stats.zscore(
                    df[column_name], nan_policy='omit'))
                df['outlier'] = z_scores > self.threshold
            else:
                # In case column is missing in the DataFrame
                df['outlier'] = False

    def plot_outliers(self, title, column_name='value'):
        """
        Plots all the data, highlighting the outliers for each dataset.
        Args:
            column_name (str): The column name based on which outliers are plotted.
            title (str): The title for the plot.
        """
        plt.figure(figsize=(14, 7))
        # Generate distinct colors for each frame
        colors = sns.color_palette(n_colors=len(self.data_frames))
        for (name, df), color in zip(self.data_frames.items(), colors):
            if column_name in df.columns:
                sns.lineplot(
                    x=df.index, y=df[column_name], data=df, label=f'{name}', color=color)
                outliers = df[df['outlier']]
                sns.scatterplot(x=outliers.index, y=column_name, data=outliers, color=color,
                                label=f'{name} Outliers', s=50, marker='o', edgecolor='black')
        plt.title(f"Outlier Detection: {title}")
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.legend()
        plt.tight_layout()
        sns.despine()
        plt.show()


def main():
    redis_connector = RedisConnector().get_connection()
    data_processor = _DataProcessor(redis_connector=redis_connector)

    tag = input("Enter category (consumption, production, source, storage): ")
    tags = get_key_by_tags(tag)
    data_frames = {}
    title = PLOT_TITLES_BY_CATEGORY.get(tag, "Figure: [no title]")

    for description, key in tags.items():
        print(f"Fetching and processing data for {description}")
        raw_data = data_processor.fetch_data_by_keytype(key)
        transformer = DataTransformer(raw_data)
        df = transformer.transform_to_dataframe()
        data_frames[description] = df

    detector = OutlierDetector(data_frames, threshold=3)
    detector.detect_outliers('value')
    detector.plot_outliers(column_name='value', title=title)


if __name__ == "__main__":
    main()
