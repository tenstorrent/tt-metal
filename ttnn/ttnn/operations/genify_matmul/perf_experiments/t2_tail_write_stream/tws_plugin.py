# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""t2_tail_write_stream pytest plugin (load with `-p tws_plugin`, this dir on PYTHONPATH): runs ANY genify_matmul
test against this dir's kernel copies (pd.KERNEL_DIR monkeypatched; the real op files are untouched) plus extra
defines for the tail write stream variants.
  TWS_DEFINES  "NAME=val+NAME2=val"  appended to the reader(s) + writer kernels (after the tail split defines)
  TWS_REAL=1   the op's real kernels dir (sanity check of the copy)
  TWS_LAYOUT=1 print the bank-local layout: logical core -> virtual NoC coords, B column class, units
"""
import os
from pathlib import Path

import pytest

HERE = Path(__file__).parent


@pytest.fixture(autouse=True)
def _tws_patch(monkeypatch):
    from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

    if os.environ.get("TWS_REAL") != "1":
        monkeypatch.setattr(pd, "KERNEL_DIR", HERE / "kernels")
    orig_tail = pd._attach_tail_split

    def _tail(descriptor, readers, writer, *a, **kw):
        orig_tail(descriptor, readers, writer, *a, **kw)
        extra = [
            tuple(d.split("=", 1)) if "=" in d else (d, "1") for d in os.environ.get("TWS_DEFINES", "").split("+") if d
        ]
        for k in (*readers, writer):
            k.defines = list(k.defines) + extra

    monkeypatch.setattr(pd, "_attach_tail_split", _tail)
    if os.environ.get("TWS_POLICY"):
        import ttnn
        from policy import policy

        st = {}
        orig_homes2 = pd._bank_homes

        def _homes2(t):
            st["dev"] = t.device()
            return orig_homes2(t)

        monkeypatch.setattr(pd, "_bank_homes", _homes2)
        orig_cols2 = pd.bank_core_columns

        def _cols2(geom, grid, gx):
            out = orig_cols2(geom, grid, gx)
            st["lay"] = {xy: (c0 % geom.b_banks, 0, w) for xy, (c0, w) in out.items()}
            return out

        monkeypatch.setattr(pd, "bank_core_columns", _cols2)
        orig_tail2 = pd._attach_tail_split

        def _tail2(descriptor, readers, writer, *a, **kw):
            orig_tail2(descriptor, readers, writer, *a, **kw)
            lay = st.pop("lay", None)
            out_t = a[-1] if a else kw.get("out")
            dram_out = out_t.memory_config().buffer_type == ttnn.BufferType.DRAM
            if lay is None or not os.environ.get("TWS_POLICY") or not dram_out:
                return  # not a bank-local split with a DRAM output: unchanged
            dev = st["dev"]
            if os.environ["TWS_POLICY"] == "PV":
                from policy_v import tail_policy

                virt = lambda c: (lambda v: (v.x, v.y))(dev.worker_core_from_logical_core(c))
                w0 = [virt(c) for c in ttnn.get_optimal_dram_bank_to_logical_worker_assignment(dev, ttnn.NOC.NOC_0)]
                w1 = [virt(c) for c in ttnn.get_optimal_dram_bank_to_logical_worker_assignment(dev, ttnn.NOC.NOC_1)]
                cores = {xy: (virt(ttnn.CoreCoord(*xy)), u, w) for xy, (u, _, w) in lay.items() if w}
                pol = {
                    xy: dict(nc=nc, wr_ep=wr, rd_ep=0, w=cores[xy][2])
                    for xy, (nc, wr) in tail_policy(cores, w0, w1, (17, 12), pd.TAIL_NC_PCT).items()
                }
            else:
                pol = policy(os.environ["TWS_POLICY"], pd.TAIL_NC_PCT, lay=lay)
            tab, rtab = [0] * 12, [0] * 12
            for (lx, ly), p in pol.items():
                v = dev.worker_core_from_logical_core(ttnn.CoreCoord(lx, ly))
                tab[v.y] |= (min(p["nc"], 7) | (p["wr_ep"] << 3)) << (4 * v.x)
                rtab[v.y] |= p["rd_ep"] << v.x
            import collections

            print(
                f"TWSBANK cores={len(pol)} nc_hist={sorted(collections.Counter((p['nc'], p['w']) for p in pol.values()).items())} wr_ep0={sum(p['wr_ep'] == 0 for p in pol.values())}"
            )
            defs = [("TWS_TAB", "1")]
            defs += [(f"TWS_TAB_Y{y}", f"{tab[y]:#x}ULL") for y in range(12)]
            defs += [(f"TWS_RTAB_Y{y}", f"{rtab[y]:#x}u") for y in range(12)]
            for k in (*readers, writer):
                k.defines = list(k.defines) + defs

        monkeypatch.setattr(pd, "_attach_tail_split", _tail2)
    if os.environ.get("TWS_LAYOUT") == "1":
        import ttnn

        state = {}
        orig_homes = pd._bank_homes

        def _homes(t):
            state["dev"] = t.device()
            return orig_homes(t)

        monkeypatch.setattr(pd, "_bank_homes", _homes)
        orig_cols = pd.bank_core_columns

        def _cols(geom, grid, gx):
            out = orig_cols(geom, grid, gx)
            dev = state.get("dev")
            for (x, y), (c0, w) in sorted(out.items(), key=lambda kv: (kv[0][1], kv[0][0])):
                p = dev.worker_core_from_logical_core(ttnn.CoreCoord(x, y)) if dev else None
                print(
                    f"LAYOUT logical=({x},{y}) virt=({p.x if p else -1},{p.y if p else -1}) col0={c0} cls={c0 % geom.b_banks} w={w}"
                )
            print("LAYOUT bank_home", geom.bank_home)
            return out

        monkeypatch.setattr(pd, "bank_core_columns", _cols)


@pytest.fixture(autouse=True)
def _tws_grad(monkeypatch):
    """TWS_GRAD=1: the op runs the graduation candidate -- grad/pd_grad.py (the real descriptor + the patch) with
    grad/kernels -- in place of the real descriptor (GENIFY_MATMUL_KNOBS also land on pd_grad)."""
    if os.environ.get("TWS_GRAD") != "1":
        return
    import ast
    import importlib
    import importlib.util

    op = importlib.import_module("ttnn.operations.genify_matmul.genify_matmul")
    from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

    spec = importlib.util.spec_from_file_location("tws_pd_grad", HERE / "grad" / "pd_grad.py")
    import sys

    g = sys.modules.get("tws_pd_grad")
    if g is None:
        g = importlib.util.module_from_spec(spec)
        sys.modules["tws_pd_grad"] = g  # dataclasses resolve their module
        spec.loader.exec_module(g)
    g.KERNEL_DIR = HERE / "grad" / "kernels"
    for name in dir(pd):  # share the exception classes the op catches
        if name.endswith("Error") and isinstance(getattr(pd, name), type):
            setattr(g, name, getattr(pd, name))
    for item in filter(None, (t.strip() for t in os.environ.get("GENIFY_MATMUL_KNOBS", "").split(","))):
        k, v = item.split("=", 1)
        setattr(g, k, ast.literal_eval(v))
    monkeypatch.setattr(op, "create_program_descriptor", g.create_program_descriptor)


@pytest.fixture(autouse=True)
def _tws_bitcheck(monkeypatch):
    """TWS_BITCHECK=1: every genify_matmul call runs twice on NaN-poisoned DRAM -- the variant, then the op's real
    kernels with no TWS defines / policy -- and the two outputs must match bit for bit (prints TILES ...)."""
    if os.environ.get("TWS_BITCHECK") != "1":
        return
    import torch
    import ttnn
    from eval.golden_tests.genify_matmul import axes
    from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

    raw = axes._raw

    def _poisoned(*args, **kwargs):
        a, b = args[0], args[1]
        dev = a.device()
        shape = [a.shape[i] for i in range(len(a.shape) - 1)] + [b.shape[len(b.shape) - 1]]
        ps = [
            ttnn.from_torch(torch.full(shape, float("nan")), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            for _ in range(2)
        ]
        for p in ps:
            p.deallocate()
        out = raw(*args, **kwargs)
        ttnn.synchronize_device(dev)
        return out

    def _checked(*args, **kwargs):
        out = _poisoned(*args, **kwargs)
        got = ttnn.to_torch(out).float()
        saved = {k: os.environ.pop(k, None) for k in ("TWS_DEFINES", "TWS_POLICY")}
        kd = pd.KERNEL_DIR
        pd.KERNEL_DIR = pd.Path(pd.__file__).parent / "kernels"
        import importlib as _il

        op = _il.import_module("ttnn.operations.genify_matmul.genify_matmul")

        cpd = op.create_program_descriptor
        op.create_program_descriptor = pd.create_program_descriptor
        try:
            ref = ttnn.to_torch(_poisoned(*args, **kwargs)).float()
        finally:
            pd.KERNEL_DIR = kd
            op.create_program_descriptor = cpd
            os.environ.update({k: v for k, v in saved.items() if v is not None})
        g2, r2 = got.reshape(-1, got.shape[-2], got.shape[-1]), ref.reshape(-1, ref.shape[-2], ref.shape[-1])
        bad = ~((g2 == r2) | (g2.isnan() & r2.isnan()))
        tiles = sorted({(bb, r // 32, c // 32) for bb, r, c in bad.nonzero().tolist()})
        print(
            f"TILES bad={len(tiles)} nan_got={int(got.isnan().sum())} nan_ref={int(ref.isnan().sum())} first={tiles[:10]}"
        )
        return out

    monkeypatch.setattr(axes, "_raw", _checked)
