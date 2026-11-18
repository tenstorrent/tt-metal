# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""pytest plugin (-p hxw_plugin, this dir on PYTHONPATH) for the t2_raw_kwalk_hxw domain sweep over the op's own perf
files, without touching them:
  HXW_SWAP=1      genify_matmul's compute kernel comes from HXW_KDIR (default ./kernels; ./grad/test_kdir = the graduation)
  HXW_DEFINES     '+'-joined defines appended to every kernel (e.g. GM_HXW+GM_HXW_STREAM)
Always prints one HXWPLAN line per op call: test id, compute subblock / K-blocks / epilogue / residency / defines."""
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
_current = {"id": "?"}


class _ComputeSwap:
    def __init__(self, real):
        self.real = real

    def __truediv__(self, name):
        kdir = Path(os.environ.get("HXW_KDIR", str(HERE / "kernels")))
        return kdir / name if name == "genify_matmul_compute.cpp" else self.real / name


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _current["id"] = item.nodeid.split("::")[-1]


@pytest.fixture(autouse=True)
def _hxw_swap(monkeypatch):
    from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd
    import importlib

    gm = importlib.import_module("ttnn.operations.genify_matmul.genify_matmul")

    if os.environ.get("HXW_SWAP") == "1":
        monkeypatch.setattr(pd, "KERNEL_DIR", _ComputeSwap(pd.KERNEL_DIR))
    extra = os.environ.get("HXW_DEFINES", "")
    if extra:
        monkeypatch.setattr(pd, "ABLATE_DEFINES", "+".join(d for d in [pd.ABLATE_DEFINES, extra] if d))
    orig = gm.create_program_descriptor

    def _traced(*args, **kwargs):
        desc = orig(*args, **kwargs)
        try:
            for k in desc.kernels:
                if str(k.kernel_source).endswith("genify_matmul_compute.cpp"):
                    ct = list(k.compile_time_args)
                    defs = sorted({d[0] for d in k.defines})
                    print(
                        f"HXWPLAN {_current['id']} sb={ct[7]}x{ct[8]} blk={ct[16]}x{ct[12]} nkb={ct[10]} "
                        f"epi={ct[13]} wres={ct[22]} ares={ct[23]} defs={'+'.join(defs)} src={Path(str(k.kernel_source)).parent.name}"
                    )
        except Exception as e:  # tracing only
            print(f"HXWPLAN {_current['id']} trace-failed {e}")
        return desc

    monkeypatch.setattr(gm, "create_program_descriptor", _traced)
