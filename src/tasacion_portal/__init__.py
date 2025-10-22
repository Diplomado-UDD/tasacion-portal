"""
Tasacion Portal - Property Price Analysis
A complete ML pipeline for real estate price prediction
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Make main modules easily importable
from . import scraper
from . import process_data
from . import train_models
from . import explain_model
from . import generate_report

__all__ = [
    "scraper",
    "process_data",
    "train_models",
    "explain_model",
    "generate_report",
]
