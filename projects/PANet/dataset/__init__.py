from .isaid import register_isaid
from .dota import register_dota
from detectron2.evaluation import RotatedCOCOEvaluator
#from .isaid import (
#    register_isaid,
#    isaid_mapper
#)
register_isaid()
register_dota()

from .dota_mapper import DotaMapper
from .isaid_mapper import ISAIDMapper
from .isaid_evaluation import ISAIDEvaluator

data_dict = {'dota1.5': {'mapper': DotaMapper, 'evaluator': RotatedCOCOEvaluator},
             'isaid': {'mapper': ISAIDMapper, 'evaluator': ISAIDEvaluator}}

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]