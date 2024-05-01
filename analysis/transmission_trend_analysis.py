from pre_processing.redis_con import RedisConnector
from analysis.trend_analysis import TrendAnalysis
from pre_processing.key_mappings import get_key_by_tags, PLOT_TITLES_BY_CATEGORY


class TransmissionTrendAnalysisIncoming(TrendAnalysis):

    def __init__(self, redis_connector):
        super().__init__(redis_connector)

    def perform_analysis(self):
        """
        Performs the complete analysis by fetching, transforming, and analyzing data based on user input.
        """
        # analysis_type = input(
        #     "Choose analysis type ('biweekly', 'monthly', 'specific'): ").strip().lower()

        # if analysis_type not in ['biweekly', 'monthly', 'specific']:
        #     print(
        #         "Invalid input. Please choose 'biweekly', 'monthly', or 'specific'.")
        #     return
        print("Default Analysis Type: Biweekly [15D]")

        tag = input(
            "Enter category (transmission_sweden, transmission_poland, transmission_latvia): ")
        tags = get_key_by_tags(tag)
        data_frames = {}
        title = PLOT_TITLES_BY_CATEGORY.get(tag, "Figure: [no title]")
        month_year_str = None

        for description, key in tags.items():
            print(f"Fetching and processing data for {description}")
            df = self.fetch_and_transform_data(key)
            analyzed_df = self.analyze_trends(df, '15D')
            # if analysis_type == 'biweekly':
            #     analyzed_df = self.analyze_trends(df, '15D')
            # elif analysis_type == 'monthly':
            #     analyzed_df = self.analyze_trends(df, 'ME')
            # elif analysis_type == 'specific':
            #     if month_year_str is None:
            #         month_year_str = input(
            #             "Enter the month and year (MM-YYYY): ")
            #         months_range = int(input(
            #             "Enter the range of months to include before and after the specified month (1-3): "))
            #     analyzed_df = self.analyze_monthly_trends(
            #         df, month_year_str, months_range)

            data_frames[description] = analyzed_df

        self.plot_trends(data_frames, title, f"{tag.capitalize()} (MW)")


if __name__ == "__main__":
    redis_connector = RedisConnector().get_connection()
    trans_trend_analysis = TransmissionTrendAnalysisIncoming(redis_connector)
    trans_trend_analysis.perform_analysis()
