# iSAID Dataset
from .isaid import register_isaid
from .dota import register_dota
#from .isaid import (
#    register_isaid,
#    isaid_mapper
#)
register_isaid()
register_dota()

from .dataset_mapper import DatasetMapper
from .dota_mapper import DotaMapper
from .isaid_evaluation import ISAIDEvaluator

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]