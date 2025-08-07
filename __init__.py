"""
Marketing Attribution Models - Advanced Multi-Touch Attribution Framework
"""

__version__ = "1.0.0"
__author__ = "Marketing Attribution Team"
__description__ = "Advanced multi-touch attribution & ROI prediction framework for enterprise marketing"

from . import attribution_engine
from . import journey_analytics
from . import roi_prediction
from . import dashboard
from . import advanced_analytics
from .config import get_config

__all__ = [
    'attribution_engine',
    'journey_analytics', 
    'roi_prediction',
    'dashboard',
    'advanced_analytics',
    'get_config'
]