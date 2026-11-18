# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""pytest plugin (-p ss_plugin, this dir on PYTHONPATH) for t2p2_swiglu_stream_epilogue over the op's own perf files,
without touching them:
  SS_SWAP=1        genify_matmul's compute kernel comes from SS_KDIR (default ./kernels); reader / writer stay real
  SS_DEFINES       '+'-joined defines appended to every kernel (e.g. GM_SS_STREAM)
  SS_COMPARE=1     every op call runs TWICE: the real op (real kernels, no extra define) then the swapped variant; the
                   two outputs are compared (torch.equal / max abs diff / max bf16 ulp) and one SSEQ line printed.
                   The test then sees the variant's output (its own CHECK prints the variant's PCC). Correctness runs
                   only (the profile then holds two GenericOps per call).
Always prints one SSPLAN line per compute-kernel descriptor: test id, subblock / block / K-blocks / epilogue / defines."""
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
_current = {"id": "?"}


class _ComputeSwap:
    def __init__(self, real):
        self.real = real

    def __truediv__(self, name):
        kdir = Path(os.environ.get("SS_KDIR", str(HERE / "kernels")))
        return kdir / name if name == "genify_matmul_compute.cpp" else self.real / name


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _current["id"] = item.nodeid.split("::")[-1]


def _ulp_bf16(a, b):
    import torch

    def key(x):
        i = x.to(torch.bfloat16).view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)

    return (key(a) - key(b)).abs().max().item()


@pytest.fixture(autouse=True)
def _ss_swap(monkeypatch, request):
    from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd
    import importlib

    gm = importlib.import_module("ttnn.operations.genify_matmul.genify_matmul")
    real_kdir = pd.KERNEL_DIR
    extra = os.environ.get("SS_DEFINES", "")
    state = {"cand": True}

    def _set_variant(cand):
        state["cand"] = cand

    orig = gm.create_program_descriptor

    def _traced(*args, **kwargs):
        # applied at call time so the perf files' own knobs (GENIFY_MATMUL_KNOBS, incl. ABLATE_DEFINES) compose
        saved_kdir, saved_abl = pd.KERNEL_DIR, pd.ABLATE_DEFINES
        if state["cand"]:
            if os.environ.get("SS_SWAP") == "1":
                pd.KERNEL_DIR = _ComputeSwap(real_kdir)
            if extra:
                pd.ABLATE_DEFINES = "+".join(d for d in [saved_abl, extra] if d)
        try:
            desc = orig(*args, **kwargs)
        finally:
            pd.KERNEL_DIR, pd.ABLATE_DEFINES = saved_kdir, saved_abl
        try:
            for k in desc.kernels:
                if str(k.kernel_source).endswith("genify_matmul_compute.cpp"):
                    ct = list(k.compile_time_args)
                    defs = sorted({d[0] for d in k.defines})
                    print(
                        f"SSPLAN {_current['id']} sb={ct[7]}x{ct[8]} blk={ct[16]}x{ct[12]} nkb={ct[10]} bb={ct[4]} "
                        f"epi={ct[13]} nat={ct[19]} wres={ct[22]} ares={ct[23]} defs={'+'.join(defs)} "
                        f"src={Path(str(k.kernel_source)).parent.parent.name}"
                    )
        except Exception as e:  # tracing only
            print(f"SSPLAN {_current['id']} trace-failed {e}")
        return desc

    monkeypatch.setattr(gm, "create_program_descriptor", _traced)

    if os.environ.get("SS_COMPARE") == "1" and hasattr(request.module, "genify_matmul"):
        import ttnn

        real_op = request.module.genify_matmul

        def _both(*args, **kwargs):
            import torch

            _set_variant(False)
            base = real_op(*args, **kwargs)
            b = ttnn.to_torch(base).float()
            base.deallocate()
            _set_variant(True)
            out = real_op(*args, **kwargs)
            c = ttnn.to_torch(out).float()
            nz = (b != c).sum().item()
            print(
                f"SSEQ {_current['id']} equal={torch.equal(b, c)} ndiff={nz}/{b.numel()} "
                f"maxabs={(b - c).abs().max().item():.3e} max_ulp_bf16={_ulp_bf16(b, c)}"
            )
            return out

        monkeypatch.setattr(request.module, "genify_matmul", _both)
