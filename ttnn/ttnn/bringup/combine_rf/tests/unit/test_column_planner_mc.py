# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The column multi-core planner (combine_planner_prepare(axis=0, cores=...), deepseek_v41_flash_d_p P.11) vs the
one-core column planner on the same random routings: every output - counts, regions, the whole token index and
hand-off table (padding and the zero tail included), the combine plan and, with det, the accumulators' order pages -
must be bit-identical, per chip (both mesh columns: id offsets 0 and Eg). The outputs are pre-filled with garbage so
a word either planner fails to write shows. Shapes: the 2x2 model's (K 6, 24 experts a chip, 9 accumulators, two
links), home tokens per chip S = 512 and 1024 (one pass), 1024 in passes of 512 and of 342 (the last pass partial),
uint32 and uint16 ids, uniform / hot / one-column routings, 32 and 8 requested cores (8: fewer cores than
accumulators, owners of two order pages)."""

import time

import pytest
import torch

import ttnn
from ttnn.bringup._testing.mesh import MESH_PARAMS
from ttnn.bringup.combine_rf.combine_plan import combine_planner_prepare, det_page_bytes, port_luts

K, EPC, A, L = 6, 24, 9, 2
GARBAGE = 0x5A5A5A5A


def _host(t, n):
    return [ttnn.to_torch(d).to(torch.int64).reshape(-1)[:n] & 0xFFFFFFFF for d in ttnn.get_device_tensors(t)]


def _fill(mesh_device, t):
    """Overwrite a planner output with garbage (every word must be rewritten by the call)."""
    R, C = tuple(mesh_device.shape)
    sh = list(t.shape)
    h = torch.full([R * sh[0], C * sh[1]] + sh[2:], GARBAGE, dtype=torch.int32)
    ht = ttnn.from_torch(
        h,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(0, 1)),
    )
    ttnn.copy_host_to_device_tensor(ht, t)


def _routing(kind, R, C, S_full, Eg, g):
    """[R, S_full, K] global ids (distinct within a token) of the column's tokens and their bf16 weights."""
    E = C * Eg
    if kind == "uniform":
        ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(R * S_full)])
    elif kind == "hot":  # a few hot experts (Zipf-like): long rows lists, slices over many cores
        w = 1.0 / torch.arange(1, E + 1, dtype=torch.float64) ** 1.2
        w = w[torch.randperm(E, generator=g)]
        ids = torch.stack([torch.multinomial(w, K, replacement=False, generator=g) for _ in range(R * S_full)])
    else:  # "column": every token's experts in one column (half the tokens column 0, half column 1)
        ids = torch.stack([(i % C) * Eg + torch.randperm(Eg, generator=g)[:K] for i in range(R * S_full)])
    w = (torch.rand(R * S_full, K, generator=g) + 0.25).to(torch.bfloat16)
    return ids.reshape(R, S_full, K), w.reshape(R, S_full, K)


@pytest.mark.timeout(2400)
@MESH_PARAMS
@pytest.mark.parametrize(
    "S_full, S",
    [(1024, 1024), (512, 512), (1024, 512), (1024, 342)],
    ids=["S1024", "S512", "S1024x512", "S1024x342"],
)
@pytest.mark.parametrize("det", [True, False], ids=["det", "arrival"])
@pytest.mark.parametrize("ncores", [32, 8], ids=["c32", "c8"])
@pytest.mark.parametrize("idw", [4, 2], ids=["u32", "u16"])
def test_column_planner_mc(mesh_device, device_params, S_full, S, det, ncores, idw):
    if idw == 2 and (ncores != 32 or S_full != S):
        pytest.skip("uint16 ids: one pass, 32 cores")
    R, C = tuple(mesh_device.shape)
    Eg = R * EPC
    passes = -(-S_full // S)
    rows_cap = -(-(R * S * min(K, EPC) + EPC * 31) // 32) * 32
    luts, nps = port_luts(mesh_device, L, per_direction=True, axis=0)
    detp = dict(rps=32, max_e=32, gu_nreg=2, pin=1, pin_ratio=8) if det else None
    grid = mesh_device.compute_with_storage_grid_size()
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    planners = []
    for p in range(passes):
        kw = dict(
            S=S,
            K=K,
            EPC=EPC,
            A=A,
            L=L,
            port_luts=luts,
            n_ports=nps,
            rows_cap=rows_cap,
            axis=0,
            id_offset=lambda r, c: c * Eg,
            tokens=S_full,
            token_base=p * S,
            ids_uint32=idw == 4,
            tile_in=True,
            det=detp,
        )
        one = combine_planner_prepare(mesh_device, core=(0, 0), **kw)
        mc = combine_planner_prepare(mesh_device, cores=[(i % grid.x, i // grid.x) for i in range(ncores)], **kw)
        planners.append((one, mc))
    M, CT = planners[0][1].mcc
    print(f"COLUMN_PLANNER_MC S_full={S_full} S={S} det={det} requested={ncores}: M={M} cores, chunk {CT} tokens")
    keys = [("counts", Eg), ("regions", Eg), ("token_index", rows_cap), ("table", 2 * rows_cap), ("plan", 1024)]
    if det:
        keys.append(("dord", A * det_page_bytes(S, K) // 4))  # (the accumulators' pages; the rest is never written)
    for kind, seed in (("uniform", 1), ("hot", 2), ("column", 3), ("uniform", 4)):
        g = torch.Generator().manual_seed(seed)
        ids, w = _routing(kind, R, C, S_full, Eg, g)
        ids_d = ttnn.from_torch(
            ids.reshape(1, 1, R * S_full, K).to(torch.int32),
            dtype=ttnn.uint32 if idw == 4 else ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=rep,
        )
        w_d = ttnn.from_torch(
            w.reshape(1, 1, R * S_full, K),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=rep,
        )
        for p, (one, mc) in enumerate(planners):
            got = {}
            for name, pl in (("one", one), ("multi", mc)):
                for k_, n in keys:
                    _fill(mesh_device, pl.outs[k_])
                    if seed == 1 and p == 0:  # (the fill itself works: else a skipped write would read as zeros)
                        assert all(bool((h == GARBAGE).all()) for h in _host(pl.outs[k_], n)), k_
                outs = pl(ids_d, w_d)
                got[name] = {k_: _host(outs[k_], n) for k_, n in keys}
            for ci in range(R * C):
                for k_, n in keys:
                    a, b = got["one"][k_][ci], got["multi"][k_][ci]
                    nd = int((a != b).sum())
                    assert nd == 0, (
                        f"{kind} seed {seed} pass {p} chip {ci} {k_}: {nd} of {n} words differ "
                        f"(first at {int((a != b).nonzero()[0])}: one {int(a[(a != b).nonzero()[0]])} "
                        f"multi {int(b[(a != b).nonzero()[0]])})"
                    )
                assert int(got["one"]["plan"][ci][600]) > 0, "no rows planned (vacuous)"
                assert not bool((got["one"]["token_index"][ci] == GARBAGE).any()), "one-core left garbage"
        print(f"COLUMN_PLANNER_MC {kind} seed {seed}: {passes} pass(es) x {R * C} chips bit-identical")
        if seed != 4:
            ttnn.deallocate(ids_d)
            ttnn.deallocate(w_d)
    # (rough wall time per call of pass 0, host dispatch included; device time: the model's tt/perf_experts.py)
    for name, i in (("one", 0), ("multi", 1)):
        pl = planners[0][i]
        pl(ids_d, w_d)
        ttnn.synchronize_device(mesh_device)
        t0 = time.time()
        for _ in range(20):
            pl(ids_d, w_d)
        ttnn.synchronize_device(mesh_device)
        print(f"COLUMN_PLANNER_MC_TIME {name} S={S} det={det} cores={ncores}: {(time.time() - t0) / 20 * 1e6:.0f} us")
