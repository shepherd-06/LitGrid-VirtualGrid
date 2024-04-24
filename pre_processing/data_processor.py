import os
import json
from datetime import datetime
from pre_processing.redis_con import RedisConnector
from pre_processing.key_mappings import get_redis_key_base


class _DataProcessor:
    def __init__(self, dataset_dir='../dataset', redis_connector=None):
        self.dataset_dir = dataset_dir
        self.redis = redis_connector if redis_connector else RedisConnector().get_connection()

    def _convert_to_redis_key(self, filename):
        # Mapping the file prefix to the desired key structure
        return get_redis_key_base(filename)

    def _process_file(self, filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
            total = 0
            missing = 0
            print("file ", filepath)
            for entry in data:
                redis_key = self._convert_to_redis_key(
                    os.path.basename(filepath), )

                if not self.redis.hexists(redis_key, entry['ltu']):
                    total += 1
                    self.redis.hset(redis_key,  mapping={
                                    entry['ltu']: entry['value']})
                else:
                    missing += 1
                    print(
                        f"Key {redis_key} already exists in Redis, skipping insertion {entry['value']}.")
            print(f"Total inserted: {total}, Missed: {missing}")

    def run(self):
        for filename in os.listdir(self.dataset_dir):
            if filename.endswith('.json'):
                print("----------------------------------")
                self._process_file(os.path.join(self.dataset_dir, filename))
                print("----------------------------------")

    def fetch_data_by_keytype(self, key_type):
        """
        Fetch all data from Redis for a given key type.
        Args:
            key_type (str): The type of data to fetch (e.g., 'electricity_consumption_actual').

        Returns:
            dict: A dictionary with Redis keys as keys and the corresponding hash values as values.
        """
        return self.redis.hgetall(key_type)


# if __name__ == "__main__":
    # redis_connector = RedisConnector().get_connection()
    # data_processor = _DataProcessor(redis_connector=redis_connector)
    # data_processor.run()
    # data_processor.fetch_data_by_keytype("wind_farms_production")
