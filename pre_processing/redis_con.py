import redis


class RedisConnector:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(RedisConnector, cls).__new__(cls)
        return cls._instance

    def __init__(self, host='localhost', port=6379, db=0):
        if not hasattr(self, '_connection'):
            self._connection = redis.StrictRedis(host=host, port=port, db=db)

    def get_connection(self):
        return self._connection


# Example usage:
# Create an instance of RedisConnector
redis_connector = RedisConnector()

# Get the Redis connection
redis_connection = redis_connector.get_connection()

# Now you can use the redis_connection object to interact with Redis
# For example:
# redis_connection.set('key', 'value')
# value = redis_connection.get('key')
