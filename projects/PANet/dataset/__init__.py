# DOTA
from .dota import register_dota
from .dota_mapper import DOTAMapper
from .dota_evaluation import DOTAEvaluator

# iSAID
from .isaid import register_isaid
from .isaid_mapper import ISAIDMapper
from .isaid_evaluation import ISAIDEvaluator

register_isaid()
register_dota()

data_dict = {'dota1.5': {'mapper': DOTAMapper, 'evaluator': DOTAEvaluator},
             'isaid': {'mapper': ISAIDMapper, 'evaluator': ISAIDEvaluator}}

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]