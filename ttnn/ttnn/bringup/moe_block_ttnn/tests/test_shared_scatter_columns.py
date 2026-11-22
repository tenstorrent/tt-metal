# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MoEBlockConfig.shared_scatter_columns + router_canonical_sum (deepseek_v41_flash_d_p PB.7, DS41-339 / DS41-341):
the moe_head's shared expert reduced by columns and the router's TP logits partials summed in chip order, so a token
row's outputs (the shared expert's row, its routing row: ids and gate weights) do not depend on which TP chip owns it.

The check (`check_rows_moved`): the head on x and on x with every SP row's rows rolled by `shift` rows (rows cross TP
chips: a chip owns S = rows / TP of them); each token's shared-expert row must be bit-identical in both (off: the rows
scatter adds a row's partials in its destination chip's ring order, so the moved rows differ - the reproduction);
plus determinism (two calls, same bytes), a padded chunk (actual_start / actual_end: each SP row's real rows a
prefix) whose real rows must equal the unpadded call's, and the output against torch (PCC; the option changes only
the add order).

Device: a mesh with TP >= 2 and 2D fabric (the model's mesh through its resident runner:
models/demos/deepseek_v41_flash_d_p/tests/test_moe_head_scatter_columns.py, BATCH_SHARED)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn


def _mb():
    return ttnn.bringup.moe_block


@dataclass(frozen=True)
class Dims:
    """DeepSeek-V4.1-Flash's MoE (checkpoint inference/config.json)."""

    E: int = 384
    K: int = 6
    H: int = 5120
    I: int = 2304
    Is: int = 2304
    route_scale: float = 1.5
    limit: float = 10.0

    def dims(self):
        mb = _mb()
        act = mb.ExpertActivation(kind=mb.ExpertActivationKind.ClampedSilu, limit=self.limit)
        return mb.MoEModelDims(
            num_experts=self.E,
            top_k=self.K,
            emb_dim=self.H,
            routed_expert_hidden_dim=self.I,
            routed_expert_activation=act,
            router=mb.RouterSpec(score_func=mb.RouterScoreFunc.SqrtSoftplus, route_scale=self.route_scale),
            shared_expert_hidden_dim=self.Is,
            shared_expert_activation=act,
        )


DS41 = Dims()


def _q(w, dtype):
    return ttnn.to_torch(ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()


def _shared_ref(d: Dims, w, x):
    b8 = ttnn.bfloat8_b
    g, u, dn = _q(w["gate"], b8), _q(w["up"], b8), _q(w["down"], b8)
    a, b = x @ g.T, x @ u.T
    return (torch.nn.functional.silu(a.clamp(max=d.limit)) * b.clamp(-d.limit, d.limit)) @ dn.T


def _weights(d: Dims, seed: int):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, sc=1.0: torch.randn(*s, generator=g) * sc  # noqa: E731
    return {
        "gate": r(d.Is, d.H, sc=d.H**-0.5),
        "up": r(d.Is, d.H, sc=d.H**-0.5),
        "down": r(d.H, d.Is, sc=d.Is**-0.5),
        "router": r(d.E, d.H, sc=d.H**-0.5),
        "bias": r(d.E, sc=0.01),
    }


class Head:
    """One MoE head (its own MoEBlockResources) on the mesh at `chunk` tokens."""

    def __init__(self, mesh, chunk: int, columns: bool, w, d: Dims = DS41):
        mb = _mb()
        self.mesh, self.chunk, self.columns, self.d = mesh, chunk, columns, d
        self.sp, self.tp = tuple(mesh.shape)
        self.dtypes = mb.MoEDtypes()
        self.dtypes.routed_weights = ttnn.bfloat4_b
        self.dtypes.shared_fidelity = ttnn.MathFidelity.LoFi
        cfg = mb.MoEBlockConfig(chunk_tokens=chunk, num_links=2, persist_l1_small=True)
        cfg.combine_snap = 2.0**-15
        cfg.shared_scatter_columns = columns
        cfg.router_canonical_sum = (
            columns  # (the router's TP logits sum in chip order: the routing rows order-free too)
        )
        self.res = mb.MoEBlockResources(mesh, d.dims(), cfg, self.dtypes)
        try:
            plan = mb.moe_block_plan(
                self.res,
                dtypes=self.dtypes,
                accumulators=None,
                max_tokens_per_expert=max(chunk, 256),
                layout=mb.FlatLayoutOptions(weights_nk=True),
                combine=mb.CombineConfig(),
            )
            self.A = plan.accumulators
        except Exception as e:  # noqa: BLE001
            logger.info(f"no block plan ({str(e).splitlines()[0][:160]}): accumulators 11")
            self.A = 11
        fit = mb.plan_moe_head(self.res, self.dtypes, self.A, None)
        assert fit.ok, fit.reason
        self.row_blocks = int(fit.row_blocks)
        lay = fit.planned[0].layout
        logger.info(
            f"moe_head chunk {chunk} columns={columns}: GY {lay.engine_rows} MT {lay.row_tiles_per_core} "
            f"NCH {lay.row_chunks} GX {lay.engine_columns}, row blocks {self.row_blocks}, A {self.A}"
        )
        f32 = lambda t: None if t is None else ttnn.from_torch(t, dtype=ttnn.float32)  # noqa: E731
        self.tw = mb.prepare_moe_head_weights(
            f32(w["gate"]),
            f32(w["up"]),
            f32(w["down"]),
            None,
            f32(w["router"]),
            f32(w["bias"]),
            resources=self.res,
            dtypes=self.dtypes,
            accumulators=self.A,
        )

    def span(self, real: list[int]) -> dict:
        """A padded chunk: SP row r holds real[r] real rows, a prefix (per device (0, r W + real[r]), the model's
        tt/packed_device.MoEPadSpan)."""
        w = self.chunk // self.sp
        words = lambda v: ttnn.from_torch(  # noqa: E731
            torch.tensor(v, dtype=torch.int64).reshape(len(v), 1, 1, 1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
        )
        ends = [r * w + real[r] for r in range(self.sp) for _ in range(self.tp)]
        return dict(actual_start=words([0] * len(ends)), actual_end=words(ends))

    def se(self, x_host: torch.Tensor, span: dict | None = None) -> torch.Tensor:
        """x [chunk, H] bf16 -> the shared expert's output [chunk, H] in token order (host)."""
        x = ttnn.from_torch(
            x_host.reshape(1, 1, *x_host.shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=(self.sp, self.tp), dims=(2, 3)),
        )
        out = _mb().moe_head(x, self.tw, resources=self.res, dtypes=self.dtypes, accumulators=self.A, **(span or {}))
        y = out.shared_expert_output
        per_chip = tuple(ttnn.get_device_tensors(y)[0].shape)
        if self.columns:
            assert per_chip[-1] == self.d.H // self.tp, f"columns scatter: y a chip {per_chip}"
            host = ttnn.to_torch(
                y, mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh, mesh_shape=(self.sp, self.tp), dims=(2, 3))
            ).reshape(self.chunk, -1)
        else:
            assert per_chip[-1] == self.d.H, f"rows scatter: y a chip {per_chip}"
            S = self.chunk // (self.sp * self.tp)
            host = torch.cat([ttnn.to_torch(t).reshape(-1, t.shape[-1])[:S] for t in ttnn.get_device_tensors(y)])
        K = self.d.K
        S = self.chunk // (self.sp * self.tp)
        rows = torch.cat(
            [ttnn.to_torch(t).reshape(-1, 2 * K).to(torch.int64)[:S] for t in ttnn.get_device_tensors(out.routing_rows)]
        )
        ttnn.deallocate(x)
        self.routing = rows.clone()  # (token order: each token's K ids, then its K gate weights' bf16 bits)
        return host.clone()

    def close(self):
        self.res.release()


def _roll_rows(x: torch.Tensor, sp: int, shift: int) -> torch.Tensor:
    """Every SP row's rows rolled by `shift` (token t of an SP row -> row (t + shift) mod rows)."""
    return torch.roll(x.reshape(sp, -1, x.shape[-1]), shifts=shift, dims=1).reshape(x.shape)


def check_rows_moved(mesh, chunk: int, columns: bool, shift: int = 32, seed: int = 7) -> dict:
    """The head on x and on x rolled by `shift` rows in every SP row: per token, rolled vs not (max abs, rows that
    differ); determinism; PCC vs torch. Returns the numbers; asserts nothing."""
    d = DS41
    w = _weights(d, seed)
    g = torch.Generator().manual_seed(1000 + seed)
    x = torch.randn(chunk, d.H, generator=g).to(torch.bfloat16)
    sp = tuple(mesh.shape)[0]
    head = Head(mesh, chunk, columns, w, d)
    W = chunk // sp
    # (padded: SP row r keeps W - r 32 - 16 real rows (> 0), a prefix crossing the TP chips' blocks)
    real = [max(16, W - 32 * r - 16) for r in range(sp)]
    try:
        a = head.se(x)
        ra = head.routing
        b = head.se(_roll_rows(x, sp, shift))
        rb = head.routing
        a2 = head.se(x)
        p = head.se(x, head.span(real))
    finally:
        head.close()
    keep = torch.zeros(chunk, dtype=torch.bool)
    for r in range(sp):
        keep[r * W : r * W + real[r]] = True
    pad_diff = float((a.float()[keep] - p.float()[keep]).abs().max())
    b_back = _roll_rows(b, sp, -shift)
    routing_rows_differ = int((ra != _roll_rows(rb, sp, -shift)).any(-1).sum())
    diff = (a.float() - b_back.float()).abs()
    rows_differ = int((diff.amax(-1) > 0).sum())
    ref = _shared_ref(d, w, x.float())
    pcc = float(torch.corrcoef(torch.stack([ref.reshape(-1), a.float().reshape(-1)]))[0, 1])
    out = {
        "chunk": chunk,
        "columns": columns,
        "row_blocks": head.row_blocks,
        "moved_max_abs_diff": float(diff.max()),
        "moved_rows_differ": rows_differ,
        "moved_routing_rows_differ": routing_rows_differ,
        "determinism_max_abs_diff": float((a.float() - a2.float()).abs().max()),
        "padded_real_rows_max_abs_diff": pad_diff,
        "pcc": pcc,
        "out": a,
    }
    logger.info(
        f"moe_head shared_scatter_columns={columns} chunk {chunk} shift {shift}: moved rows max abs "
        f"{out['moved_max_abs_diff']} ({rows_differ} of {chunk} rows differ; routing rows {routing_rows_differ}), determinism "
        f"{out['determinism_max_abs_diff']}, padded chunk's real rows vs unpadded {pad_diff}, pcc vs torch {pcc:.6f}, "
        f"row blocks {head.row_blocks}"
    )
    return out


def check_scatter_columns(mesh, chunks=(2048, 8192), max_grid=None) -> list[dict]:
    """The regression: off reproduces (moved rows differ), on is bit-identical, deterministic, accurate, one row block;
    on vs off within a few bf16 ulps (only the add order differs). max_grid (x, y): the block's core cap
    (set_max_grid), set before the first plan - a process that later runs a model with a cap must get the same one
    (the cap is fixed at a process's first plan)."""
    if tuple(mesh.shape)[1] < 2:
        pytest.skip("needs TP >= 2")
    if max_grid is not None:
        _mb().set_max_grid(*max_grid)
    results = []
    off = check_rows_moved(mesh, chunks[0], columns=False)
    results.append(off)
    assert off["moved_rows_differ"] > 0, "rows scatter: moved rows identical - the reproduction did not reproduce"
    assert off["determinism_max_abs_diff"] == 0.0
    assert off["padded_real_rows_max_abs_diff"] == 0.0, "rows scatter: a padded chunk's real rows differ"
    for chunk in chunks:
        on = check_rows_moved(mesh, chunk, columns=True)
        results.append(on)
        assert on["moved_max_abs_diff"] == 0.0, f"columns scatter chunk {chunk}: moved rows differ"
        assert on["moved_routing_rows_differ"] == 0, f"canonical router sum chunk {chunk}: moved routing rows differ"
        assert on["determinism_max_abs_diff"] == 0.0, f"columns scatter chunk {chunk}: not deterministic"
        assert on["padded_real_rows_max_abs_diff"] == 0.0, f"columns scatter chunk {chunk}: padded real rows differ"
        assert on["pcc"] > 0.999, f"columns scatter chunk {chunk}: pcc {on['pcc']}"
        assert on["row_blocks"] == 1, f"columns scatter chunk {chunk}: {on['row_blocks']} row blocks"
        if chunk == off["chunk"]:
            dd = (on["out"].float() - off["out"].float()).abs()
            scale = off["out"].float().abs().max()
            logger.info(f"on vs off chunk {chunk}: max abs {float(dd.max())} (|y| max {float(scale)})")
            assert float(dd.max()) <= float(scale) * 2.0**-6, "columns vs rows scatter: more than the add order"
    return results


@pytest.mark.timeout(1800)
def test_shared_scatter_columns(mesh_device):
    check_scatter_columns(mesh_device)
