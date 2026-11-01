# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Point the op at this experiment's descriptor/kernels copy (variant/). Also a pytest plugin (-p swap)."""

import importlib.util
import sys
from pathlib import Path

from types import SimpleNamespace

from ttnn.operations.layer_norm_tile import layer_norm as _layer_norm

_spec = importlib.util.spec_from_file_location(
    "srm_variant_pd", Path(__file__).parent / "variant" / "layer_norm_tile_program_descriptor.py"
)
pd = importlib.util.module_from_spec(_spec)
sys.modules["srm_variant_pd"] = pd
_spec.loader.exec_module(pd)

# ttnn may hold a second copy of the op module; patch the globals the public entry actually resolves.
_globals = _layer_norm.__globals__
_globals["create_program_descriptor"] = pd.create_program_descriptor
_globals["select_blocking"] = pd.select_blocking


class _Globals(SimpleNamespace):
    def __setattr__(self, name, value):
        _globals[name] = value


op_module = _Globals()
