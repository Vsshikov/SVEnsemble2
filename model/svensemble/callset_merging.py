"""
This module contains functions for merging variant callsets from different sources.

The callsets are presumed to be in breakpoint (BP) notation. See `caller_specific.py` and `bndbpconv.py` for more details.
"""

import copy
import importlib
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
import pyranges as pr
from scipy.spatial import cKDTree
import os

from . import vcfparse as vcfp
from . import bndbpconv as bndc 
from . import utils as sv_utils


