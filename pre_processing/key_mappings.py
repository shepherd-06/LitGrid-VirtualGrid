KEY_STRUCTURE = {
    'Actual National Electricity Consumption': 'electricity_consumption_actual',
    'Actual national electricity generation': 'electricity_generation',
    'Actual national production of wind farms': 'wind_farms_production',
    'Actual national solar generation': 'solar_generation',
    'Actual national production of hydroelectric power plants': 'hydroelectric_production',
    'Actual national production of storage devices': 'storage_devices_production',
    'Planned national electricity consumption': 'electricity_consumption_planned',
    'Planned national electricity production': 'electricity_production_planned',
    'Projected national electricity consumption': 'electricity_consumption_projected',
    'Actual production of thermal power plants connected to PT': 'thermal_plants_production',
    'Actual national production of other energy sources': 'other_sources_production'
}


def get_redis_key_base(filename):
    # Removes common extensions and replaces underscores with spaces
    base = filename.replace('_', ' ').replace('.json', '').strip()
    # Remove possible suffixes that could interfere with matching
    base = base.replace(' (MW)', '').strip()
    # Map filename base to the Redis key base
    return KEY_STRUCTURE.get(base, 'unknown_data_type')
