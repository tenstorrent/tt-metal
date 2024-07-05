import sys
import os
from pathlib import Path

# load _ttnn.so
import _ttnn


def get_ttnn_module():
    print(dir(_ttnn))
    return _ttnn


def get_tt_lib_module():
    return _ttnn.experimental
