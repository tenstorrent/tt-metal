# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pkgutil
import sys
from importlib.util import module_from_spec

__all__ = []

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    __all__.append(module_name)
    spec = loader.find_spec(module_name)
    _module = module_from_spec(spec)
    # Register the module in sys.modules before executing it
    sys.modules[f"{module_name}"] = _module
    spec.loader.exec_module(_module)
    globals()[module_name] = _module
