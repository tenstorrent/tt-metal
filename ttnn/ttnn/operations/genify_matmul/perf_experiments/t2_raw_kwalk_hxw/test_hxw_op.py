# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""t2_raw_kwalk_hxw whole-op bench: the real genify_matmul with the compute kernel swapped for this dir's copy
(KERNEL_DIR monkeypatched so the compute kernel comes from ./kernels; reader / writer stay the real ones), per variant:

  base       the real op (real kernels dir)
  kdir       this dir's kernels, no define (must be bit-identical to base)
  ref        helper Interm walk + identity copy (GM_HXW_REF_INTERM) -- bit-identity reference of `a`
  a          raw h x w walk + identity copy (GM_HXW + GM_HXW_NONE_FORCE)
  b          raw h x w walk, last K-block streamed per row group (a + GM_HXW_STREAM)
  hxw        GM_HXW only (the host NONE_RAW_WALK gate decides the epilogue-none path; epilogue walks raw)
  hxw_b      GM_HXW + GM_HXW_STREAM (host gate decides)
  grad       the graduation patch's compute (grad/genify_matmul_compute.cpp + grad/genify_matmul_kwalk.hpp)

Env: HXW_CASES="file:case,..." (file in r16 r11 r15 flux guard sharded; default r16:te_gate), HXW_VARIANTS
(default base,a,b), plus the perf dir's GENIFY_MATMUL_KNOBS / FORCE / BLOCK / PLAN / CHECK. Every variant's output
is compared with the first variant's (torch.equal) and, with GENIFY_MATMUL_CHECK=1, with torch (PCC)."""
import importlib.util
import math
import os
from pathlib import Path

import pytest
import ttnn

HERE = Path(__file__).resolve().parent
PERF = HERE.parents[5] / "tests/ttnn/unit_tests/operations/genify_matmul/perf"

VARIANTS = {
    "base": (False, []),
    "kdir": (True, []),
    "ref": (True, ["GM_HXW_REF_INTERM"]),
    "a": (True, ["GM_HXW", "GM_HXW_NONE_FORCE"]),
    "b": (True, ["GM_HXW", "GM_HXW_NONE_FORCE", "GM_HXW_STREAM"]),
    "hxw": (True, ["GM_HXW"]),
    "hxw_b": (True, ["GM_HXW", "GM_HXW_STREAM"]),
    "grad": ("grad", []),  # the graduation (grad/ files; the host NONE_RAW_WALK gate decides as today)
}


class _ComputeSwap:
    """KERNEL_DIR stand-in: this dir's compute kernel, the real reader / writer (their zone source paths unchanged)."""

    def __init__(self, real, kdir):
        self.real, self.kdir = real, kdir

    def __truediv__(self, name):
        return self.kdir / name if name == "genify_matmul_compute.cpp" else self.real / name


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_hxw_{name}", PERF / f"test_genify_matmul_perf_{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cases():
    return [c for c in os.environ.get("HXW_CASES", "r16:te_gate").split(",") if c]


@pytest.mark.parametrize("sel", _cases())
def test_hxw_op(sel, device, monkeypatch, request):
    import torch  # local: the ttnn tree forbids a global torch import
    from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

    fname, cname = sel.split(":")
    r11 = _load("r11")
    cases = (
        {c[0]: c for c in getattr(_load(fname), f"{fname.upper()}_CASES")}
        if fname != "r11"
        else {c[0]: c for c in r11.R11_CASES}
    )
    case = cases[cname]

    class _Req:  # _apply_knobs reads request.node.callspec.id for GENIFY_MATMUL_ONLY / PLAN labels
        class node:
            class callspec:
                id = cname

    monkeypatch.delenv("GENIFY_MATMUL_ONLY", raising=False)
    r11._apply_knobs(monkeypatch, _Req)
    _, m, k, n, w_dt, fid, epilogue, *_ = case
    torch.manual_seed(0)
    A = torch.randn(1, m, k)
    B = torch.randn(k, n) / math.sqrt(k)
    from eval.golden_tests.genify_matmul.axes import observed as genify_matmul
    from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor

    ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tb = create_ttnn_input_tensor(B, device, dtype=getattr(ttnn, w_dt), layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, fid),
        fp32_dest_acc_en=os.environ.get("HXW_FP32", "1") == "1",
        math_approx_mode=False,
    )
    kwargs = {"activation": "swiglu"} if epilogue == "swiglu" else {}
    ref = None
    if os.environ.get("GENIFY_MATMUL_CHECK") == "1":
        ref = A.float() @ ttnn.to_torch(tb).float()
        if epilogue == "swiglu":
            t = ref.reshape(1, m, n // 64, 2, 32)
            ref = (torch.nn.functional.silu(t[..., 0, :]) * t[..., 1, :]).reshape(1, m, n // 2)
    base_kdir, base_abl = pd.KERNEL_DIR, pd.ABLATE_DEFINES
    first = None
    prev = []
    for v in [x for x in os.environ.get("HXW_VARIANTS", "base,a,b").split(",") if x]:
        own, defs = VARIANTS[v]
        kdir = HERE / ("grad/test_kdir" if own == "grad" else "kernels")
        monkeypatch.setattr(pd, "KERNEL_DIR", _ComputeSwap(base_kdir, kdir) if own else base_kdir)
        monkeypatch.setattr(pd, "ABLATE_DEFINES", "+".join([d for d in [base_abl, *defs] if d]))
        out = genify_matmul(ta, tb, compute_kernel_config=cfg, **kwargs)
        ttnn.synchronize_device(device)
        got = ttnn.to_torch(out).float()
        line = f"HXW {sel} variant={v}"
        if first is None:
            first = (v, got)
        else:
            line += f" maxdiff_vs_{first[0]}={(got - first[1]).abs().max().item():.3e}"
        line += " equal=[" + ",".join(pv for pv, po in prev if torch.equal(po, got)) + "]"
        prev.append((v, got))
        if ref is not None:
            pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
            rms = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
            line += f" pcc={pcc:.6f} rel_rms={rms:.5f}"
            assert pcc > 0.999, line
        print(line)
        out.deallocate()
