"""
Model architectures for drug response prediction
"""
try:
    from .deep_learning import ImprovedDrugResponseModel
except ImportError:
    from deep_learning import ImprovedDrugResponseModel

__all__ = ['ImprovedDrugResponseModel']
