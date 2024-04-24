import unittest
import sys
import os
from pre_processing.redis_con import RedisConnector

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestRedisConnector(unittest.TestCase):
    def test_redis_connection(self):
        # Initialize RedisConnector
        redis_connector = RedisConnector()

        # Get the Redis connection
        redis_connection = redis_connector.get_connection()

        # Assert that the Redis connection is not None
        self.assertIsNotNone(redis_connection)

        # Test some basic Redis commands
        redis_connection.set('test_key', 'test_value')
        self.assertEqual(redis_connection.get('test_key'), b'test_value')


if __name__ == '__main__':
    unittest.main()
