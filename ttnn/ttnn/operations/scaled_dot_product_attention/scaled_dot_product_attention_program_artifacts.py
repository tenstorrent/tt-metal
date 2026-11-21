# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for scaled_dot_product_attention on Quasar (op_design.md).

Work unit (a "NEO unit"): q_chunk_tiles query tile-rows x Dt of one query head of one sequence. The units of
one (sequence b, key/value head c) are ordered q-chunk-major, u -> (q_chunk = u // g, member = u % g), query
head h = c * g + member. A round is C consecutive units of one (b, c): NEO t of a cluster computes unit
head_round * C + t, and every NEO of the cluster shares the round's keys and values through ALL rings (read
once per cluster per round). The last round of a head is padded: a NEO without a unit recomputes NEO 0's unit
(the reader feeds it the same tiles and the writer rewrites the same pages with the same bytes), so every
NEO takes the same number of entries on every ring. Whole rounds are dealt to clusters.

Per round the keys are walked in chunks of k_chunk_tiles tile-rows (online softmax: running max, running sum,
running output, rescaled when the max grows). With one chunk (k_chunk_tiles == Skt, the resident regime) the
rescale never runs; with several (the streamed regime) it does. Both regimes run the same kernels.

Thread plan (R, C, W): R reader harts, C NEOs, W writer harts per cluster, from the one constant THREAD_PLAN.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# ---------------------------------------------------------------------------
# Knobs (op_design.md, Blocking Model): each defined once; everything else derives from them.
# ---------------------------------------------------------------------------

# The thread plan (R readers, C NEOs, W writers): a compute-bound op that reads more than it writes.
THREAD_PLAN = (4, 4, 2)
# Q blocks per NEO: the next round's Q lands while this round computes.
Q_BUFFER_DEPTH = 2
# K / V blocks per producer region of the ALL rings: R blocks in flight cluster-wide.
KV_BLOCKS_PER_READER = 1
# Output tiles per NEO: normalize/pack overlaps the writer's DRAM write.
OUT_BUFFER_DEPTH = 2
# L1 held back from the ring budget (the unreserved-size query already excludes the reserved region).
L1_RESERVE_BYTES = 0
# A coarse q_chunk_tiles must not starve the key chunk below this many tile-rows.
K_CHUNK_FLOOR_TILES = 4

# The cluster model of the target page (.claude/references/targets/quasar.md); Python has no query for them.
NEOS = 4
FREE_DM_HARTS = 6
ALL_CONSUMER_FANOUT = 4
DFB_MAX_RING_BYTES = 1_048_560  # one ring's compute-side limit (1 MiB - 16 B)

TILE = 32



def const_tiles(causal: bool) -> int:
    """Constant tiles per NEO in cb_constants: the reduce scaler, plus the diagonal mask when causal (their entry
    indices are SCALER_TILE / DIAG_MASK_TILE in kernels/sdpa_geometry.hpp). Key tiles entirely above the diagonal
    are filled with -inf in DEST, so the design's third "full mask" tile is not needed."""
    return 2 if causal else 1


def validate_thread_plan(threads):
    r, c, w = threads
    if not (
        1 <= c <= min(NEOS, ALL_CONSUMER_FANOUT)
        and r >= 1
        and w >= 1
        and r + w <= FREE_DM_HARTS
        and c % r == 0
        and c % w == 0
    ):
        raise ValueError(
            f"thread plan (R, C, W) = {threads}: need C <= {min(NEOS, ALL_CONSUMER_FANOUT)}, "
            f"R + W <= {FREE_DM_HARTS}, R | C, W | C"
        )


def _ceil(a, b):
    return -(-a // b)


def _tile_bytes(dtype):
    return {ttnn.bfloat16: 2048, ttnn.float32: 4096}[dtype]


@dataclasses.dataclass(frozen=True)
class SdpaPlan:
    R: int
    C: int
    W: int
    B: int
    Hq: int
    Hkv: int
    g: int
    Sqt: int
    Skt: int
    Dt: int
    qc: int  # q_chunk_tiles: query tile-rows per NEO unit
    kc: int  # k_chunk_tiles: key tile-rows per chunk
    d_block_tiles: int  # head-dim tiles per block (pinned to Dt in this phase)
    causal: bool
    in_dtype: object
    acc_dtype: object
    out_dtype: object
    num_clusters: int
    budget: int

    @property
    def nqc(self):
        return _ceil(self.Sqt, self.qc)

    @property
    def units_per_head(self):
        return self.g * self.nqc

    @property
    def rounds_per_head(self):
        return _ceil(self.units_per_head, self.C)

    @property
    def total_rounds(self):
        return self.B * self.Hkv * self.rounds_per_head

    @property
    def num_k_chunks(self):
        return _ceil(self.Skt, self.kc)

    @property
    def streamed(self):
        return self.kc < self.Skt

    @property
    def T_in(self):
        return _tile_bytes(self.in_dtype)

    @property
    def T_acc(self):
        return _tile_bytes(self.acc_dtype)

    @property
    def T_out(self):
        return _tile_bytes(self.out_dtype)

    @property
    def const_tiles(self):
        return const_tiles(self.causal)

    def ring_entries(self):
        return ring_entries(
            R=self.R, C=self.C, qc=self.qc, kc=self.kc, Dt=self.d_block_tiles, causal=self.causal,
            in_dtype=self.in_dtype, acc_dtype=self.acc_dtype, out_dtype=self.out_dtype,
        )  # fmt: skip

    def l1_bytes(self):
        return ring_bytes(self.ring_entries())


def ring_entries(*, R, C, qc, kc, Dt, causal, in_dtype, acc_dtype, out_dtype):
    """{ring: (entries, dtype)}: STRIDED rings count entries per NEO x C, ALL rings per producer x R. The one
    statement of every ring's capacity: the program's DataflowBufferSpecs, the L1 total and the fit predicate
    all evaluate it (l1_ledger.md)."""
    return {
        "query_block": (C * Q_BUFFER_DEPTH * qc * Dt, in_dtype),
        "key_block": (R * KV_BLOCKS_PER_READER * kc * Dt, in_dtype),
        "value_block": (R * KV_BLOCKS_PER_READER * kc * Dt, in_dtype),
        # The scaler and the mask in the scores' format: the row-max reduce moves DEST into SrcB, whose
        # format is the scaler's.
        "constants": (C * const_tiles(causal), acc_dtype),
        "scores": (C * qc * kc, acc_dtype),
        "probs": (C * qc * kc, acc_dtype),
        "row_max": (C * qc, acc_dtype),
        "rescale": (C * qc, acc_dtype),
        "row_sum": (C * qc, acc_dtype),
        "out_accum": (C * qc * Dt, acc_dtype),
        "out": (C * OUT_BUFFER_DEPTH, out_dtype),
    }


def ring_bytes(rings):
    return sum(n * _tile_bytes(dt) for n, dt in rings.values())


def _fits(rings, budget):
    """Every ring under the part's ring limit, and all of them together under the cluster's L1 budget."""
    return all(n * _tile_bytes(dt) <= DFB_MAX_RING_BYTES for n, dt in rings.values()) and ring_bytes(rings) <= budget


def _kc_fit(qc, Skt, budget, **geometry):
    """The coarsest key chunk (<= Skt) whose rings fit at this q chunk; 0 when none does. Every ring's bytes are
    non-decreasing in kc, so the fitting chunks are a prefix of 1..Skt: bisect it."""
    lo, hi = 0, Skt
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _fits(ring_entries(qc=qc, kc=mid, **geometry), budget):
            lo = mid
        else:
            hi = mid - 1
    return lo


def make_plan(q, k, *, is_causal, fp32_dest_acc_en, threads=THREAD_PLAN, q_chunk_tiles=None, k_chunk_tiles=None):
    """The blocking plan of one call: the fit predicate of op_design.md (Work Distribution). Knob overrides
    (tests) are clamped to legality and never exceed the fit."""
    validate_thread_plan(threads)
    R, C, W = threads
    B, Hq, S, D = (int(d) for d in q.shape)
    Hkv = int(k.shape[1])
    Sqt, Dt = _ceil(S, TILE), _ceil(D, TILE)
    Skt = Sqt  # the keys have the queries' tokens (X4)
    grid = q.device().compute_with_storage_grid_size()
    num_clusters = grid.x * grid.y
    budget = ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES
    acc_dtype = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
    geometry = dict(R=R, C=C, Dt=Dt, causal=bool(is_causal), in_dtype=q.dtype, acc_dtype=acc_dtype, out_dtype=q.dtype)
    fit = lambda qc_: _kc_fit(qc_, Skt, budget, **geometry)  # noqa: E731

    # Occupancy first (every NEO of the grid a unit), then the coarsest unit that leaves the key chunk its floor.
    # kc_fit is 0 when a ring that does not depend on kc (Q, O) already breaks a limit, so the loop also walks
    # qc down past those.
    qc = min(max(B * Hq * Sqt // (C * num_clusters), 1), Sqt)
    if q_chunk_tiles is not None:
        qc = min(max(int(q_chunk_tiles), 1), Sqt)
    while qc > 1 and fit(qc) < min(Skt, K_CHUNK_FLOOR_TILES):
        qc -= 1
    kc = fit(qc)
    if k_chunk_tiles is not None:
        kc = min(max(int(k_chunk_tiles), 1), kc)
    if kc < 1:
        raise NotImplementedError(
            f"scaled_dot_product_attention: head dimension D={D} needs the wide-head-dim regime "
            f"(its blocks do not fit one cluster's L1 at q_chunk_tiles = k_chunk_tiles = 1)"
        )
    return SdpaPlan(
        R=R, C=C, W=W, B=B, Hq=Hq, Hkv=Hkv, g=Hq // Hkv, Sqt=Sqt, Skt=Skt, Dt=Dt, qc=qc, kc=kc,
        d_block_tiles=Dt, causal=bool(is_causal), in_dtype=q.dtype, acc_dtype=acc_dtype, out_dtype=q.dtype,
        num_clusters=num_clusters, budget=budget,
    )  # fmt: skip


def split_rounds(total_rounds, grid_size):
    """(core set, {core: rounds}, {core: first round}): whole rounds dealt to clusters, row-major."""
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(
        grid_size, total_rounds, row_wise=True
    )
    counts, starts, start = {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            counts[core], starts[core] = per_core, start
            start += per_core
    return all_cores, counts, starts


def _compute_hw_config(compute_config, plan):
    # Every Float32 ring the compute kernel reads is unpacked to the source registers (the FPU matmul, reduce
    # and broadcast operations read it there); bfloat16 rings keep the default.
    unpack_modes = {
        name: E.UnpackMode.UnpackToSrc
        for name, (_, dtype) in plan.ring_entries().items()
        if dtype == ttnn.float32 and name != "out"
    }
    return E.ComputeHardwareConfig(
        # Set explicitly on every compute kernel.
        fpu_math_fidelity=compute_config.math_fidelity,
        sfpu_precision_mode=E.Precision.Approximate if compute_config.math_approx_mode else E.Precision.Precise,
        enable_32_bit_dest=bool(compute_config.fp32_dest_acc_en),
        unpack_modes=unpack_modes,
    )


def create_program_artifacts(q, k, v, out, *, plan: SdpaPlan, scale_bits, compute_config, tag_threads=False):
    """(ProgramSpec, ProgramRunArgs, tensor indices) of one call."""
    p = plan
    core_set, counts, starts = split_rounds(p.total_rounds, q.device().compute_with_storage_grid_size())

    dfbs = [
        E.DataflowBufferSpec(unique_id=name, entry_size=_tile_bytes(dtype), num_entries=n, data_format_metadata=dtype)
        for name, (n, dtype) in p.ring_entries().items()
    ]
    tensor_parameters = [
        E.TensorParameter(unique_id=n, spec=t.spec) for n, t in (("q", q), ("k", k), ("v", v), ("out", out))
    ]
    geometry = {
        "reader_threads": p.R,
        "compute_threads": p.C,
        "writer_threads": p.W,
        "Hq": p.Hq,
        "Hkv": p.Hkv,
        "group": p.g,
        "Sqt": p.Sqt,
        "Skt": p.Skt,
        "Dt": p.d_block_tiles,
        "q_chunk_tiles": p.qc,
        "k_chunk_tiles": p.kc,
        "units_per_head": p.units_per_head,
        "rounds_per_head": p.rounds_per_head,
        "causal": int(p.causal),
        "const_tiles": p.const_tiles,
    }
    rounds_schema = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["first_round", "num_rounds"])

    reader_hw = E.create_reader_datamovement_config()
    # The reader keeps the credits of every ring it fills (whole blocks, assembled from several reads, and
    # the constant tiles it writes itself): explicit sync, every transfer tagged with a user id.
    reader_hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(
        disable_dfb_implicit_sync_for=["query_block", "key_block", "value_block", "constants"]
    )
    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "scaled_dot_product_attention_reader.cpp"),
        hw_config=reader_hw,
        num_threads=p.R,
        dfb_bindings=[
            E.producer_of("query_block", "query_block"),
            E.producer_of("key_block", "key_block"),
            E.producer_of("value_block", "value_block"),
            E.producer_of("constants", "constants"),
        ],
        tensor_bindings=[
            E.TensorBinding(tensor_parameter_name="q", accessor_name="q"),
            E.TensorBinding(tensor_parameter_name="k", accessor_name="k"),
            E.TensorBinding(tensor_parameter_name="v", accessor_name="v"),
        ],
        compile_time_args=geometry,
        runtime_arg_schema=rounds_schema,
    )
    internal = ["scores", "probs", "row_max", "rescale", "row_sum", "out_accum"]
    compute = E.KernelSpec(
        unique_id="compute",
        source=str(KERNEL_DIR / "scaled_dot_product_attention_compute.cpp"),
        hw_config=_compute_hw_config(compute_config, p),
        num_threads=p.C,
        dfb_bindings=[
            E.consumer_of("query_block", "query_block"),
            E.all_consumer_of("key_block", "key_block"),  # every NEO of the cluster reads every K block
            E.all_consumer_of("value_block", "value_block"),
            E.consumer_of("constants", "constants"),
            *[b for name in internal for b in (E.producer_of(name, name), E.consumer_of(name, name))],
            E.producer_of("out", "out"),
        ],
        compile_time_args={**geometry, "tag_threads": int(bool(tag_threads))},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=["first_round", "num_rounds"], common_runtime_arg_names=["scale_bits"]
        ),
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "scaled_dot_product_attention_writer.cpp"),
        hw_config=E.create_writer_datamovement_config(),
        num_threads=p.W,
        dfb_bindings=[E.consumer_of("out", "out")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="out", accessor_name="out")],
        compile_time_args=geometry,
        runtime_arg_schema=rounds_schema,
    )
    spec = E.ProgramSpec(
        name="scaled_dot_product_attention",
        kernels=[reader, compute, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "compute", "writer"], target_nodes=core_set)],
    )
    rounds = {"first_round": starts, "num_rounds": counts}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader", runtime_arg_values=rounds),
            E.KernelRunArgs(
                kernel="compute", runtime_arg_values=rounds, common_runtime_arg_values={"scale_bits": scale_bits}
            ),
            E.KernelRunArgs(kernel="writer", runtime_arg_values=rounds),
        ]
    )
    return spec, run_args, {"q": 0, "k": 1, "v": 2, "out": 3}
