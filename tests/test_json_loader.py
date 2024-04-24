import unittest
import sys
import os
import json
from pre_processing.json_loader import JSONLoader

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestJSONLoader(unittest.TestCase):
    def test_load_json_from_parent_folder(self):
        # Create a temporary JSON file for testing
        test_data = {'key': 'value'}
        test_file_path = 'test_file.json'
        with open(test_file_path, 'w') as file:
            json.dump(test_data, file)

        # Initialize JSONLoader with the test file name
        json_loader = JSONLoader(test_file_path)

        # Load JSON content from the parent folder
        json_content = json_loader.load_json_from_parent_folder()

        # Remove the temporary JSON file
        os.remove(test_file_path)

        # Assert that JSON content matches the test data
        self.assertEqual(json_content, test_data)

    def test_load_json_from_parent_folder_not_found(self):
        # Initialize JSONLoader with a non-existent file name
        json_loader = JSONLoader('non_existent_file.json')

        # Load JSON content from the parent folder
        json_content = json_loader.load_json_from_parent_folder()

        # Assert that JSON content is None (file not found)
        self.assertIsNone(json_content)


if __name__ == '__main__':
    unittest.main()
