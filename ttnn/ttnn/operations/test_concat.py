# SPDX-FileCopyrightText: © 2024 BOS

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, Dict

import sys
import ttnn

test_concat = ttnn.register_operation()(ttnn._ttnn.operations.test_ops.test_concat)

__all__ = []