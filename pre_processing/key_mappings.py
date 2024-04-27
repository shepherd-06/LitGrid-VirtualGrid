KEY_STRUCTURE = {
    """
    A manually maintained dictionary that maps descriptive terms for datasets to their corresponding Redis key names and categories. 
    
    This structure is not dynamically linked to the categorization function `get_key_by_tags`. 
    Changes in this dictionary, such as adding new categories or changing descriptions, must be manually reflected in the `get_key_by_tags` function logic. Failure to update both can lead to mismatches and errors in key retrieval and categorization.

    Example:
        KEY_STRUCTURE = {
            'Actual National Electricity Consumption': 'electricity_consumption_actual', # consumption,
            ...
    }

    This approach ensures that keys are organized by their function but requires careful maintenance 
    to keep the structure and the categorization logic in sync.
    """
    'Actual National Electricity Consumption': 'electricity_consumption_actual',  # consumption
    'Actual national electricity generation': 'electricity_generation',  # production
    'Actual national production of wind farms': 'wind_farms_production',  # source
    'Actual national solar generation': 'solar_generation',  # source
    'Actual national production of hydroelectric power plants': 'hydroelectric_production',  # source
    'Actual national production of storage devices': 'storage_devices_production',  # storage
    'Planned national electricity consumption': 'electricity_consumption_planned',  # consumption
    'Planned national electricity production': 'electricity_production_planned',  # production
    'Projected national electricity consumption': 'electricity_consumption_projected',  # consumption
    'Actual production of thermal power plants connected to PT': 'thermal_plants_production',  # source
    'Actual national production of other energy sources': 'other_sources_production'  # source
}


PLOT_TITLES_BY_CATEGORY = {
    'consumption': 'Electricity Consumption Trends',
    'production': 'Electricity Production Trends',
    'source': 'Electricity Source Distribution',
    'storage': 'Electricity Storage Capacity Trends'
}


def get_redis_key_base(filename):
    # Removes common extensions and replaces underscores with spaces
    base = filename.replace('_', ' ').replace('.json', '').strip()
    # Remove possible suffixes that could interfere with matching
    base = base.replace(' (MW)', '').strip()
    # Map filename base to the Redis key base
    return KEY_STRUCTURE.get(base, 'unknown_data_type')


def get_key_by_tags(tag: str):
    """
    Retrieves Redis keys by their associated category tags from the manually maintained `KEY_STRUCTURE`. 
    This function filters the keys based on the categories predefined in the `KEY_STRUCTURE` dictionary. 
    Each category corresponds to a specific operational aspect like 
        'consumption', 'production', 'source', or 'storage'.

    Args:
        tag (str): The category tag used to filter the keys. 
        Expected values are 'consumption', 'production', 'source', or 'storage'.

    Returns:
        dict: A dictionary containing descriptions and their corresponding Redis keys that match the given tag.

    Note:
        This function is not dynamic. It relies on the static `KEY_STRUCTURE` where each change in the key or category definition must be manually updated in the dictionary. Any updates or changes in the category tags or the addition of new keys require a corresponding update in this function to maintain consistency. This manual maintenance is prone to errors and requires careful synchronization between the dictionary updates and function logic.

    Example:
        matched_keys = get_key_by_tags('consumption')  # Returns all keys associated with electricity consumption.
    """
    if tag == "consumption":
        return {
            'Actual National Electricity Consumption': 'electricity_consumption_actual',
            'Planned national electricity consumption': 'electricity_consumption_planned',
            'Projected national electricity consumption': 'electricity_consumption_projected',
        }
    elif tag == "production":
        return {
            'Planned national electricity production': 'electricity_production_planned',  # production
            'Actual national electricity generation': 'electricity_generation',  # production
        }
    elif tag == "source":
        return {
            'Actual national production of wind farms': 'wind_farms_production',  # source
            'Actual national solar generation': 'solar_generation',  # source
            'Actual national production of hydroelectric power plants': 'hydroelectric_production',  # source
            'Actual production of thermal power plants connected to PT': 'thermal_plants_production',  # source
            'Actual national production of other energy sources': 'other_sources_production'  # source
        }
    elif tag == "storage":
        return {
            'Actual national production of storage devices': 'storage_devices_production',
        }
    else:
        # return error
        return {}
