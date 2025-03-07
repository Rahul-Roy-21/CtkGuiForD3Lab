import json
from copy import deepcopy

class ConfigManager:
    def __init__(self, 
            config_path="configurable_props.json", 
            static_path="static_props.json",
            validation_path = "validation.json"
    ):
        self.config_path = config_path
        self.static_path = static_path
        self.validation_path = validation_path
        self._load_configs()
        self.sorted_setting_keys = None

    def _load_configs(self):
        with open(self.config_path, "r", encoding="utf-8") as file:
            self.config = json.load(file)
        
        with open(self.static_path, "r", encoding="utf-8") as file:
            self.static_config = json.load(file)
        
        with open(self.validation_path, "r", encoding="utf-8") as file:
            self.valid_props = json.load(file)

    def get(self, key):
        """Retrieve value from either configurable or static properties."""
        keys = key.split(".")
        
        data = self._get_nested(self.config, keys)  # Check config first
        if data is None:
            data = self._get_nested(self.static_config, keys)  # Fallback to static_config
        
        if data is None:
            raise KeyError(f"Key '{key}' not found in configuration.")
        
        return data

    def _get_nested(self, data, keys):
        """Helper function to get a nested value, returning None if missing."""
        for k in keys:
            if not isinstance(data, dict) or k not in data:
                return None
            data = data[k]
        return data

    def set_temp(self, key, value, data):
        """Modify temp_config (only configurable_props), raises KeyError if key does not exist."""
        keys = key.split(".")
        for k in keys[:-1]:
            if k not in data:
                raise KeyError(f"Key '{key}' not found in configurable properties.")
            data = data[k]
        
        if keys[-1] not in data:
            raise KeyError(f"Key '{key}' not found in configurable properties.")
        
        data[keys[-1]] = value

    def get_updated_config(self, updated_keys_map: dict):
        temp_config = deepcopy(self.config)
        for k, v in updated_keys_map.items():
            self.set_temp(k,v, temp_config)
        return temp_config
    
    def _flatten_keys(self, data, prefix=""):
        """Recursively extract all keys as dot-separated strings."""
        keys = []
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                keys.extend(self._flatten_keys(value, full_key))
            else:
                keys.append(full_key)
        return keys

    def get_sorted_setting_keys(self):
        """Return a sorted list of all keys in self.config as dot-separated strings."""
        if not self.sorted_setting_keys:
            self.sorted_setting_keys = sorted(self._flatten_keys(self.config))
        return self.sorted_setting_keys

    def get_valid_props_map(self, key):
        """Retrieve value for the dot-connected key from valid.json."""
        keys = key.split(".")
        return self._get_nested(self.valid_props, keys)

    def save_changes(self, updated_config):
        """Commit temp changes and save to configurable_props.json."""
        with open(self.config_path, "w", encoding="utf-8") as file:
            json.dump(updated_config, file, indent=4)
        
        # Reloads the config with updates
        self.config = deepcopy(updated_config)