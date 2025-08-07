"""
Configuration settings for Marketing Attribution Models
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Database configuration
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'marketing_attribution'),
    'username': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
}

# API configurations
API_CONFIGS = {
    'google_ads': {
        'developer_token': os.getenv('GOOGLE_ADS_DEVELOPER_TOKEN'),
        'client_id': os.getenv('GOOGLE_ADS_CLIENT_ID'),
        'client_secret': os.getenv('GOOGLE_ADS_CLIENT_SECRET'),
        'refresh_token': os.getenv('GOOGLE_ADS_REFRESH_TOKEN'),
    },
    'facebook': {
        'app_id': os.getenv('FACEBOOK_APP_ID'),
        'app_secret': os.getenv('FACEBOOK_APP_SECRET'),
        'access_token': os.getenv('FACEBOOK_ACCESS_TOKEN'),
    },
    'google_analytics': {
        'property_id': os.getenv('GA_PROPERTY_ID'),
        'credentials_path': os.getenv('GA_CREDENTIALS_PATH'),
    }
}

# Attribution model settings
ATTRIBUTION_CONFIG = {
    'default_lookback_window': 30,  # days
    'attribution_window': 90,  # days
    'minimum_touchpoints': 2,
    'decay_rate': 0.7,  # for time decay models
    'markov_order': 1,  # first-order Markov chain
    'shapley_sample_size': 10000,
    'confidence_level': 0.95,
}

# Model training settings
MODEL_CONFIG = {
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42,
    'cross_validation_folds': 5,
    'hyperparameter_trials': 100,
    'early_stopping_rounds': 50,
}

# Dashboard settings
DASHBOARD_CONFIG = {
    'refresh_interval': 3600,  # seconds
    'chart_height': 400,
    'color_scheme': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'export_formats': ['png', 'pdf', 'html'],
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': str(LOGS_DIR / 'attribution.log'),
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {}

def get_config(section: str = None) -> Dict[str, Any]:
    """Get configuration settings"""
    configs = {
        'database': DATABASE_CONFIG,
        'apis': API_CONFIGS,
        'attribution': ATTRIBUTION_CONFIG,
        'model': MODEL_CONFIG,
        'dashboard': DASHBOARD_CONFIG,
        'logging': LOGGING_CONFIG,
    }
    
    if section:
        return configs.get(section, {})
    return configs