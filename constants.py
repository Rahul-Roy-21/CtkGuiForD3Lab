import os
from util.config_manager import ConfigManager

MAIN_DIR = os.path.dirname(os.path.realpath(__file__))
IMG_DIR = os.path.join(MAIN_DIR, 'images')
OUT_DIR = os.path.join(MAIN_DIR, 'output')

DATA_PATH = os.path.join(MAIN_DIR, 'data')
CONFIGURABLE_DATA_PATH = os.path.join(DATA_PATH, 'configurable_props.json')
STATIC_DATA_PATH = os.path.join(DATA_PATH, 'static_props.json')
VALIDATION_JSON_PATH = os.path.join(DATA_PATH, 'validation.json')

my_config_manager = ConfigManager(
    config_path=CONFIGURABLE_DATA_PATH,
    static_path=STATIC_DATA_PATH,
    validation_path=VALIDATION_JSON_PATH
)