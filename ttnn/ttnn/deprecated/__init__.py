# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from ttnn._ttnn.deprecated import device, profiler


import ttnn.deprecated.fused_ops

"""
Clean up steps:
1 .TODO: device and profiler functions need to be moved to `ttnn/device.py` and `ttnn/profiler.py` respectively
   Please convert all of them to have pythonic names using `snake_case` instead of `PascalCase`
   Please try to match C++ API with Python API as much as possible. For example, if C++ API has `device::get_current_device()`, then Python API should have `device.get_current_device()`
   Or if C++ API has `profiler::start()`, then Python API should have `profiler.start()`
2. Delete fused_ops from ttnn.deprecated (delete all the uses first, move them to somewhere in models folder, they shouldn't be part of our public API)
3. Delete fallback_ops  from ttnn.deprecated (delete all the uses first, simply use regular torch code instead)
4. Move ttnn.deprecated._internal/comparison_funcs.py out of ttnn to models?
5. Move ttnn.deprecated.utils out of ttnn to models as well?
"""
