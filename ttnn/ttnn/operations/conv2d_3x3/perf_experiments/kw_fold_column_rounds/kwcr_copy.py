# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Load one of this experiment's op copies as a package and retarget an existing conv2d_3x3 test module at it.

    op_base/: conv2d_3x3 at e40c435b625 (the baseline), plus a host-only gate override (KWCR_FORCE_KWF=1).
    op_cand/: the same with kw-fold column rounds (kernels/conv2d_3x3_kw_fold.hpp, the weight streams, the gate).

No __init__.py in the copies: `import ttnn` must not walk them. Host-only helpers; no torch, no device.
One copy per pytest process (profiling an op next to a copy of itself can abort on zone-id hash collisions)."""

import importlib
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent


def copy_modules(copy):
    pkg_name = f"kwcr_{copy}"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(HERE / copy)]
        sys.modules[pkg_name] = pkg
    return (
        importlib.import_module(f"{pkg_name}.conv2d_3x3"),
        importlib.import_module(f"{pkg_name}.conv2d_3x3_program_descriptor"),
    )


def retarget(test_module_name, namespace, copy):
    """Import a conv2d_3x3 test module, point every name it took from the real op at the copy, and re-export its
    test_* functions into `namespace`."""
    op_mod, pd_mod = copy_modules(copy)
    real_pkg = importlib.import_module("ttnn.operations.conv2d_3x3")
    real_op = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3")
    real_pd = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor")
    m = importlib.import_module(test_module_name)
    swapped = []
    for name, value in list(vars(m).items()):
        if value is real_op:
            setattr(m, name, op_mod)
        elif value is real_pd:
            setattr(m, name, pd_mod)
        elif name in getattr(real_pkg, "__all__", ()) and value is getattr(real_pkg, name, None):
            setattr(m, name, getattr(op_mod, name))
        else:
            continue
        swapped.append(name)
    for name, value in vars(m).items():
        if name.startswith("test_") and callable(value):
            namespace[name] = value
    return swapped


# Short ids of the 22 perf focus cases, in feature_spec order (test_conv2d_3x3_perf_focus's order).
SHORT_IDS = [
    "stem1",
    "stem2",
    "stem3",
    "res2",
    "res3_0",
    "res3_x",
    "res4_0",
    "res4_x",
    "res5_0",
    "res5_1",
    "res5_2",
    "aspp1",
    "aspp2",
    "aspp3",
    "semdec_r3_fc0",
    "instdec_r3_fc0",
    "semdec_r3_fc1",
    "semdec_r2_fc0",
    "instdec_r2_fc0",
    "sem_r2_fc1",
    "inst_r2_fc1",
    "center1",
]


def focus_cases(selection):
    """The perf focus cases named in `selection` (comma list of SHORT_IDS, in that order; empty = all)."""
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    cases = [c for c in LOOSE_CASES if "[perf]" in c["extras"]["layer"]]
    assert len(cases) == len(SHORT_IDS)
    want = [s for s in selection.split(",") if s] if selection else SHORT_IDS
    unknown = set(want) - set(SHORT_IDS)
    assert not unknown, unknown
    picked = [(c, i) for c, i in zip(cases, SHORT_IDS) if i in want]
    return [c for c, _ in picked], [i for _, i in picked]
