# model/__init__.py
from .generate_data import generate_data
from .create_model import create_model
from .train_model import train_and_evaluate
from .main import weights_to_c_array, save_model_files, plot_results

__all__ = [
    'generate_data',
    'create_model', 
    'train_and_evaluate',
    'weights_to_c_array',
    'save_model_files',
    'plot_results'
]