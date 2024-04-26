import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pre_processing.redis_con import RedisConnector
from pre_processing.data_processor import _DataProcessor
from pre_processing.data_transformer import DataTransformer
from pre_processing.key_mappings import get_key_by_tags


class DistributionVisualizer:
    def __init__(self, data_frames):
        """
        Initialize the DistributionVisualizer with data.

        Args:
            data_frames (dict): A dictionary where the key is a descriptive name and the value is a DataFrame.
        """
        self.data_frames = data_frames

    def plot_histogram(self, title, column='value', bins=25):
        """
        Plots the histogram for the frequency distribution of consumption values.

        Args:
            title (str): The title for the histogram.
            column (str): The name of the column in the DataFrame to plot the histogram for.
            bins (int): The number of bins to use for the histogram.
        """
        plt.figure(figsize=(12, 6))
        for name, df in self.data_frames.items():
            sns.histplot(df[column], bins=bins, label=name,
                         kde=True, element='step')

        plt.title(title)
        plt.xlabel('Value (MW)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()


def main():
    redis_connector = RedisConnector().get_connection()
    data_processor = _DataProcessor(redis_connector=redis_connector)

    tag_mapping = {
        "1": "consumption",
        "2": "production",
        "3": "source",
        "4": "storage"
    }

    choice_visualizer = input(
        "Enter 1 for consumption, 2 for production, 3 for source, 4 for storage: ")
    if choice_visualizer not in tag_mapping:
        print("Invalid choice. Please enter a number between 1 and 4.")
        return

    choice_bin = input("Provide bin value: (default is 50) ")

    try:
        choice_bin = int(choice_bin)
    except ValueError:
        print("Invalid choice. Please enter a number between 1 and 75.")
        return

    if choice_bin < 0 or choice_bin >= 75:
        choice_bin = 50
    

    tags = get_key_by_tags(tag=tag_mapping[choice_visualizer])
    data_frames = {}

    for description, key in tags.items():
        print(f"Fetching data for {description}")
        raw_data = data_processor.fetch_data_by_keytype(key)
        transformer = DataTransformer(raw_data)
        df = transformer.transform_to_dataframe()
        data_frames[description] = df

    visualizer = DistributionVisualizer(data_frames)
    visualizer.plot_histogram(
        f'{tag_mapping[choice_visualizer].capitalize()} Electricity Distribution')


if __name__ == "__main__":
    main()
