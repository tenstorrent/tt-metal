# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""combine_planner_prepare's column / pass options (axis=0, id_offset, tokens / token_base, ids_uint32, tile_in, det)
on the mesh vs the host model plan_reference (deepseek_v41_flash_d_p P.8): per chip (r, c) of a mesh whose tokens are
replicated over the row, the column's tokens' top-k ids (global, column-major expert groups: column c holds
[c Eg, (c + 1) Eg)) and bf16 weights as TILE [R S, K]; every pass's counts / regions / token index / hand-off table /
combine plan must equal the host model bit for bit; with det, each owned token's order list holds exactly its rows'
sources in this column. Negative control: the host model with another id offset or pass base differs."""

import pytest
import torch

import ttnn
from ttnn.bringup._testing.mesh import MESH_PARAMS
from ttnn.bringup.combine_rf.combine_plan import combine_planner_prepare, plan_reference, port_luts

K, EPC, A, L = 6, 24, 9, 2


def _host(t, n):
    return [ttnn.to_torch(d).to(torch.int64).reshape(-1)[:n] & 0xFFFFFFFF for d in ttnn.get_device_tensors(t)]


@pytest.mark.timeout(1200)
@MESH_PARAMS
@pytest.mark.parametrize("S_full, passes", [(256, 2), (192, 3)], ids=["256x2", "192x3"])
@pytest.mark.parametrize("det", [False, True], ids=["arrival", "det"])
def test_column_planner(mesh_device, device_params, S_full, passes, det):
    R, C = tuple(mesh_device.shape)
    Eg = R * EPC
    S = -(-S_full // passes)
    rows_cap = -(-(R * S * K + EPC * 31) // 32) * 32
    g = torch.Generator().manual_seed(7)
    # per mesh row r its S_full tokens' top-k of all C Eg experts (distinct within a token)
    ids = torch.stack([torch.randperm(C * Eg, generator=g)[:K] for _ in range(R * S_full)]).reshape(R, S_full, K)
    w = (torch.rand(R, S_full, K, generator=g) + 0.25).to(torch.bfloat16)
    wbits = w.view(torch.int16).to(torch.int64) & 0xFFFF
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    ids_d = ttnn.from_torch(
        ids.reshape(1, 1, R * S_full, K).to(torch.int32),
        dtype=ttnn.uint32,
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
    luts, nps = port_luts(mesh_device, L, per_direction=True, axis=0)
    detp = dict(rps=32, max_e=32, gu_nreg=2, pin=1, pin_ratio=8) if det else None
    for p in range(passes):
        pl = combine_planner_prepare(
            mesh_device,
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
            ids_uint32=True,
            tile_in=True,
            det=detp,
        )
        outs = pl(ids_d, w_d)
        got = {
            k: _host(outs[k], n)
            for k, n in (("counts", Eg), ("regions", Eg), ("token_index", rows_cap), ("plan", 1024))
        }
        tabs = _host(outs["table"], 2 * rows_cap)
        dord = _host(outs["dord"], 16 * (S * K + 63) // 64 * 64 // 4) if det else None
        for ci in range(R * C):
            r, c = divmod(ci, C)
            ref = plan_reference(
                ids,
                wbits,
                r,
                R,
                1,
                S,
                K,
                EPC,
                Eg,
                A,
                L,
                nps[(r, c)],
                luts[(r, c)],
                rows_cap,
                id_offset=c * Eg,
                token_base=p * S,
            )
            for k in ("counts", "regions", "token_index", "plan"):
                assert torch.equal(got[k][ci], torch.tensor(ref[k], dtype=torch.int64)), (p, ci, k)
            # (the table's rows of the placed experts; padding rows unused)
            assert torch.equal(tabs[ci], torch.tensor(ref["table"], dtype=torch.int64)), (p, ci, "table")
            if p == 0 and ci == 0:  # negative controls: another column's offset / another pass base
                bad = plan_reference(
                    ids,
                    wbits,
                    r,
                    R,
                    1,
                    S,
                    K,
                    EPC,
                    Eg,
                    A,
                    L,
                    nps[(r, c)],
                    luts[(r, c)],
                    rows_cap,
                    id_offset=(c + 1) % C * Eg,
                    token_base=p * S,
                )
                assert C == 1 or not torch.equal(got["counts"][ci], torch.tensor(bad["counts"])), "control (offset)"
                bad = plan_reference(
                    ids,
                    wbits,
                    r,
                    R,
                    1,
                    S,
                    K,
                    EPC,
                    Eg,
                    A,
                    L,
                    nps[(r, c)],
                    luts[(r, c)],
                    rows_cap,
                    id_offset=c * Eg,
                    token_base=(p + 1) * S,
                )
                assert not torch.equal(got["counts"][ci], torch.tensor(bad["counts"])), "control (pass)"
            if det:  # every owned token's order list: exactly its rows' sources in this column, padded with 0xFF
                page = (S * K + 63) // 64 * 64
                b = dord[ci].numpy().astype("<u4").tobytes()
                for t in range(min(S, S_full - p * S)):
                    a, j = t % A, t // A
                    lst = list(b[a * page + j * K : a * page + j * K + K])
                    g_ = ids[r, p * S + t] - c * Eg
                    srcs = sorted(int(v) // EPC for v in g_ if 0 <= int(v) < Eg)
                    assert sorted(v for v in lst if v != 0xFF) == srcs, (p, ci, t, lst, srcs)
                    assert all(v == 0xFF for v in lst[len(srcs) :]), (p, ci, t, lst)
