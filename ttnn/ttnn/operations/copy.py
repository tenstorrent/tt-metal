# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, Dict

import tt_lib as ttl

import sys
import ttnn


THIS_MODULE = sys.modules[__name__]

__all__ = []

typecast = ttnn.register_operation()(ttnn._ttnn.operations.copy.typecast)

__all__ = []
