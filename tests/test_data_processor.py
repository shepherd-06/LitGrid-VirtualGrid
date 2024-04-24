import unittest
from unittest.mock import patch, MagicMock
from pre_processing.data_processor import _DataProcessor
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class Test_DataProcessor(unittest.TestCase):

    def setUp(self):
        # Mock RedisConnector and its methods
        self.mock_redis = MagicMock()
        self.mock_redis.exists.return_value = False

        # Initialize the _DataProcessor with the mocked RedisConnector
        self.processor = _DataProcessor(redis_connector=self.mock_redis)

    def test_convert_to_redis_key(self):
        # Test the private _convert_to_redis_key method
        test_cases = [
            ("Actual National Electricity Consumption (MW).json",
             "2023-01-01 02:00:00", "electricity_consumption_actual_2023010102"),
            ("Actual national electricity generation (MW).json",
             "2023-01-01 03:00:00", "electricity_generation_2023010103"),
            ("Actual national production of wind farms (MW).json",
             "2023-01-01 04:00:00", "wind_farms_production_2023010104"),
            ("Actual national solar generation (MW).json",
             "2023-01-01 05:00:00", "solar_generation_2023010105"),
            ("Actual national production of hydroelectric power plants (MW).json",
             "2023-01-01 06:00:00", "hydroelectric_production_2023010106"),
            ("Actual national production of storage devices.json",
             "2023-01-01 07:00:00", "storage_devices_production_2023010107"),
            ("Planned national electricity consumption (MW).json",
             "2023-01-01 08:00:00", "electricity_consumption_planned_2023010108"),
            ("Planned national electricity production (MW).json",
             "2023-01-01 09:00:00", "electricity_production_planned_2023010109"),
            ("Projected national electricity consumption (MW).json",
             "2023-01-01 10:00:00", "electricity_consumption_projected_2023010110"),
            ("Actual production of thermal power plants connected to PT (MW).json",
             "2023-01-01 11:00:00", "thermal_plants_production_2023010111"),
            ("Actual national production of other energy sources.json",
             "2023-01-01 12:00:00", "other_sources_production_2023010112"),
        ]

        for filename, timestamp, expected_key in test_cases:
            with self.subTest(filename=filename, timestamp=timestamp):
                result_key = self.processor._convert_to_redis_key(
                    filename, timestamp)
                self.assertEqual(result_key, expected_key)

    @patch('os.listdir')
    @patch('builtins.open')
    @patch('json.load')
    def test_run(self, mock_json_load, mock_open, mock_listdir):
        # Set up the mocks
        mock_listdir.return_value = [
            'Actual National Electricity Consumption (MW).json']
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_json_load.return_value = [
            {"id": "203", "value": 978.868, "ltu": "2023-01-01 02:00:00",
                "utc": "2023-01-01 00:00:00"},
            {"id": "204", "value": 1120.342, "ltu": "2023-01-01 03:00:00",
                "utc": "2023-01-01 01:00:00"},
            {"id": "205", "value": 1045.223, "ltu": "2023-01-01 04:00:00",
                "utc": "2023-01-01 02:00:00"},
            {"id": "206", "value": 990.564, "ltu": "2023-01-01 05:00:00",
                "utc": "2023-01-01 03:00:00"},
            {"id": "207", "value": 875.780, "ltu": "2023-01-01 06:00:00",
                "utc": "2023-01-01 04:00:00"},
        ]

        # Run the processor
        self.processor.run()

        # Check if Redis methods were called correctly
        self.mock_redis.exists.assert_called()
        self.mock_redis.hmset.assert_called_with(
            'electricity_consumption_actual_2023010102', {'value': 978.868})

        # Verify that Redis exists was called to check if the key exists
        self.assertTrue(self.mock_redis.exists.called)


if __name__ == '__main__':
    unittest.main()
