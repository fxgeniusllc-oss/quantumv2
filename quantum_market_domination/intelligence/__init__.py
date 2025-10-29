"""Intelligence module initialization"""

from .correlation_engine import CorrelationEngine
from .predictive_surface import PredictiveSurface
from .opportunity_evaluator import OpportunityEvaluator, OpportunityType

__all__ = [
    'CorrelationEngine',
    'PredictiveSurface',
    'OpportunityEvaluator',
    'OpportunityType'
]
