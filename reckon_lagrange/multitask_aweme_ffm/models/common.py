import sys
sys.path.append('../common/api')
sys.path.append('../feature')

import math
import tensorflow as tf

import lagrange as lg
import lagrange_model as lgm

FEATURE_VERSION = 'finish-v1'
from finish_feature_v1 import *