# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-op before/after for the fast Sinkhorn (E2). In-process DEVICE KERNEL DURATION, variants interleaved:
  base = the real op (ttnn/ttnn/operations/mhc_pre), grad = ../op/ (descriptor copy + patched compute kernel).
env: SKB_SHAPES="640x1792,..." SKB_DTYPES="xbf16,xf32" SKB_OPS="base,grad" SKB_KNOBS="default,own5" SKB_REPEAT=5
outputs of grad are compared bitwise with base (the Sinkhorn change is bit-identical).
"""
import os
import importlib

import pytest

torch = importlib.import_module("torch")
import ttnn

import importlib as _il

real_op = _il.import_module("ttnn.operations.mhc_pre.mhc_pre")
import ttnn.operations.mhc_pre.mhc_pre_program_descriptor as base_pd
import ttnn.operations.mhc_pre.perf_experiments.sinkhorn_sfpu_fast.op.mhc_pre_program_descriptor as grad_pd

_KEY = "DEVICE KERNEL DURATION [ns]"
_DT = {"xbf16": ttnn.bfloat16, "xf32": ttnn.float32}
MODS = {"base": base_pd, "grad": grad_pd}
KNOBS = {
    "default": {},
    "own0": dict(OWNER_C_DISCOUNT=0),
    "own3": dict(OWNER_C_DISCOUNT=3),
    "own4": dict(OWNER_C_DISCOUNT=4),
    "own5": dict(OWNER_C_DISCOUNT=5),
    "own6": dict(OWNER_C_DISCOUNT=6),
    "own7": dict(OWNER_C_DISCOUNT=7),
}


def _read_ns(device):
    ttnn.ReadDeviceProfiler(device)
    out = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for p in programs:
            e = (getattr(p, "program_analyses_results", None) or {}).get(_KEY)
            if e is not None:
                out.append(float(e.duration))
    return out


def test_op_perf(device, monkeypatch):
    shapes = [tuple(int(v) for v in s.split("x")) for s in os.environ.get("SKB_SHAPES", "640x1792").split(",")]
    dts = os.environ.get("SKB_DTYPES", "xbf16").split(",")
    ops = os.environ.get("SKB_OPS", "base,grad").split(",")
    knobs = os.environ.get("SKB_KNOBS", "default").split(",")
    rep = int(os.environ.get("SKB_REPEAT", 5))
    rows = []
    for T, C in shapes:
        for dt in dts:
            torch.manual_seed(0)
            x = torch.randn((1, 1, T, 4 * C), dtype=torch.float32)
            w = torch.randn((4 * C, 24), dtype=torch.float32) / (4 * C) ** 0.5
            b = torch.randn((1, 24), dtype=torch.float32)
            tx = ttnn.from_torch(x, dtype=_DT[dt], layout=ttnn.TILE_LAYOUT, device=device)
            tw = ttnn.from_torch(
                w, dtype=getattr(ttnn, os.environ.get("SKB_WDT", "float32")), layout=ttnn.TILE_LAYOUT, device=device
            )
            tb = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            ns = {(o, k): [] for o in ops for k in knobs}
            outs = {}
            _read_ns(device)
            for r in range(rep):
                for o in ops:
                    for k in knobs:
                        with monkeypatch.context() as m:
                            m.setattr(real_op, "create_program_descriptor", MODS[o].create_program_descriptor)
                            for name, value in KNOBS[k].items():
                                m.setattr(MODS[o], name, value)
                            y, post, comb = real_op.mhc_pre(tx, tw, tb, scale=(1.0, 1.0, 1.0), sinkhorn_iters=20)
                            ttnn.synchronize_device(device)
                            ns[(o, k)] += _read_ns(device)
                            got = [ttnn.to_torch(t).float() for t in (y, post, comb)]
                            if r == 0:
                                outs[(o, k)] = got
                            else:
                                ok = [
                                    torch.equal(a.view(torch.int32), b_.view(torch.int32))
                                    for a, b_ in zip(got, outs[(o, k)])
                                ]
                                if not all(ok):
                                    print(f"NONDET {T}x{C} {dt} {o}/{k} rep {r}: y/post/comb same = {ok}")
            ref = outs[(ops[0], knobs[0])]
            if os.environ.get("SKB_PREC"):  # precision vs fp64 Sinkhorn on dumped logits (SKB_PREC=<logits .pt prefix>)
                L = (
                    torch.load(f"{os.environ['SKB_PREC']}_{T}x{C}_{dt}.pt")[("grad", "default")][2]
                    .double()
                    .reshape(-1, 4, 4)
                )
                m = torch.softmax(L, dim=-1) + 1e-6
                m = m / (m.sum(dim=-2, keepdim=True) + 1e-6)
                for _ in range(19):
                    m = m / (m.sum(dim=-1, keepdim=True) + 1e-6)
                    m = m / (m.sum(dim=-2, keepdim=True) + 1e-6)
                m32 = torch.softmax(L.float(), dim=-1) + 1e-6
                m32 = m32 / (m32.sum(dim=-2, keepdim=True) + 1e-6)
                for _ in range(19):
                    m32 = m32 / (m32.sum(dim=-1, keepdim=True) + 1e-6)
                    m32 = m32 / (m32.sum(dim=-2, keepdim=True) + 1e-6)
                t32 = (m32.double() - m).abs()
                print(f"PREC {T}x{C} {dt} torch_fp32: maxabs {t32.max().item():.3e} meanabs {t32.mean().item():.3e}")
                for key, v in outs.items():
                    d = (v[2].double().reshape(-1, 4, 4) - m).abs()
                    print(
                        f"PREC {T}x{C} {dt} {key[0]}/{key[1]} vs fp64: maxabs {d.max().item():.3e} meanabs {d.mean().item():.3e} maxrel {(d / m.abs()).max().item():.3e}"
                    )
            for key, v in outs.items():
                print(
                    f"NANCHECK {T}x{C} {dt} {key[0]}/{key[1]}: comb NaNs {int(torch.isnan(v[2]).sum())} / {v[2].numel()}"
                )
            tag = f"{T}x{C}_{dt}"
            if os.environ.get("SKB_SAVE"):
                torch.save(outs, f"{os.environ['SKB_SAVE']}_{tag}.pt")
            if os.environ.get("SKB_REF"):
                saved = torch.load(f"{os.environ['SKB_REF']}_{tag}.pt")
                for key, v in outs.items():
                    r0 = saved[(ops[0], knobs[0])]
                    same = [torch.equal(a.view(torch.int32), b_.view(torch.int32)) for a, b_ in zip(v, r0)]
                    print(f"BITWISE {T}x{C} {dt} {key[0]}/{key[1]} vs SAVED {ops[0]}/{knobs[0]}: y/post/comb {same}")
            for key, v in outs.items():
                same = [torch.equal(a.view(torch.int32), b_.view(torch.int32)) for a, b_ in zip(v, ref)]
                dmax = [(a - b_).abs().max().item() for a, b_ in zip(v, ref)]
                print(
                    f"BITWISE {T}x{C} {dt} {key[0]}/{key[1]} vs {ops[0]}/{knobs[0]}: y/post/comb {same} maxdiff {dmax}"
                )
                if key[0] != ops[0]:
                    a, b_ = v[2].reshape(-1, 16), ref[2].reshape(-1, 16)
                    ulp = (a.view(torch.int32).long() - b_.view(torch.int32).long()).abs()
                    print(
                        f"BITWISE   comb ulp per (i,j): max {ulp.max(0).values.tolist()} count {(ulp > 0).sum(0).tolist()}"
                    )
                for name, a, b_ in zip(("y", "post", "comb"), v, ref):
                    bad = (a != b_).reshape(-1, a.shape[-1]).any(-1).nonzero().flatten().tolist()
                    if bad:
                        print(
                            f"BITWISE   {name}: {len(bad)} differing token rows, first {bad[:12]}, tile-rows {sorted(set(t // 32 for t in bad))[:20]}"
                        )
            for key, v in ns.items():
                rows.append((f"{T}x{C}", dt, key[0], key[1], v))
    for s, dt, o, k, v in rows:
        med = sorted(v)[len(v) // 2] / 1000 if v else float("nan")
        print(f"PERF {s} {dt} {o} {k} median {med:.2f}us |", " ".join(f"{e / 1000:.1f}" for e in v))
