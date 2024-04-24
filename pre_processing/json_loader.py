import os
import json

class JSONLoader:
    def __init__(self, filename):
        self.filename = filename

    def load_json_from_parent_folder(self):
        # Get the parent directory path
        parent_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        
        # Construct the path to the JSON file in the parent directory
        json_file_path = os.path.join(parent_directory, self.filename)

        # Check if the file exists
        if os.path.exists(json_file_path) and os.path.isfile(json_file_path):
            # Load and return the JSON content
            with open(json_file_path, 'r') as file:
                json_data = json.load(file)
            return json_data
        else:
            print(f"Error: File '{self.filename}' not found in the parent directory.")
            return None

# Example usage:
# Create an instance of JSONLoader with the filename
json_loader = JSONLoader('example.json')

# Load JSON content from the parent folder
json_content = json_loader.load_json_from_parent_folder()

if json_content:
    print(json_content)
