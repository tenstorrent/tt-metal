# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import pkgutil
import sys
from importlib.util import module_from_spec

# The Python ops of nomic-embed-text-v2-moe (ttnn.generic_op: the op, its program descriptor and kernels) live beside
# the model's C++ ops in ttnn/cpp/ttnn/operations/experimental/nomic_embed_text_v2_moe/<op>/ and import as
# ttnn.operations.<op>. The C++ ops' directories there have no __init__.py, so only the Python ones are loaded.
_NOMIC_OPS = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "cpp", "ttnn", "operations", "experimental", "nomic_embed_text_v2_moe"
    )
)
if os.path.isdir(_NOMIC_OPS):
    __path__.append(_NOMIC_OPS)

__all__ = []

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    __all__.append(module_name)
    spec = loader.find_spec(module_name)
    _module = module_from_spec(spec)
    # Register the module in sys.modules before executing it
    sys.modules[f"{module_name}"] = _module
    spec.loader.exec_module(_module)
    globals()[module_name] = _module
