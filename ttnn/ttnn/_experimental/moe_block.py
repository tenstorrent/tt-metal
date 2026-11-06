# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.moe_block: the MoE block's ops, config structs, shared host math and MoEBlockResources.

Everything is bound in C++ (ttnn/cpp/ttnn/operations/experimental/moe_block/). ttnn's auto-registration puts the ops
(bound with the "ttnn.experimental.moe_block." prefix) in a module of this name before this file loads; this module
replaces it, so it takes those ops over and adds the classes and functions of the nanobind submodule.
"""

import sys as _sys

import ttnn as _ttnn

_cpp = _ttnn._ttnn.operations.experimental.moe_block
_registered = _sys.modules.get("ttnn.experimental.moe_block")

for _name in dir(_cpp):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_cpp, _name)
if _registered is not None and _registered is not _sys.modules.get(__name__):
    for _name in dir(_registered):
        if not _name.startswith("_"):
            globals()[_name] = getattr(_registered, _name)  # (the registered ttnn operations win)

del _name
