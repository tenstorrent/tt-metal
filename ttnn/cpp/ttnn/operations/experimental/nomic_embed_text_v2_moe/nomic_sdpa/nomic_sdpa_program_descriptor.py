# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for nomic_sdpa (op_design.md § Circular Buffers / § Work Distribution).

Every CB page count derives from the Plan's block knobs (q_block_rows Q, key_block_tiles N, head_dim_tiles HD,
pv_out_tiles PV, the depths); the grid is the plan's first `num_cores` cores row-major. The per-core work split is
the common-RT `core_row_start` table, indexed in-kernel by core_index = y * 11 + x. The only per-core runtime args are
the K/V multicast roles (kv_multicast_topology; reader and writer), fixed for a cache key.
"""

import os
from pathlib import Path

import ttnn

import importlib

from .nomic_sdpa import BF16_TILE_BYTES, GRID, HEADS, _scale_bits

# Knobs are read from the op module at build time (single source; a sweep can patch them).
_op = importlib.import_module(".nomic_sdpa", __package__)

KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices (names binding, indices free; op_design.md § Circular Buffers)
CB_QUERY = 0
CB_KEY_T = 1
CB_VALUE_AUG = 2
CB_SCORES = 3
CB_ROW_STATS = 4
# 5 was cb_pv_out ([O | l] round trip; gone with the normalize in DEST)
CB_OUT = 6
CB_REDUCE_SCALER = 7
CB_PROBS = 8  # exp_overlap only: P of one row group (the scores ring keeps only scores)

# Slots in the cached descriptor that bind_call() patches per call
READER_SLOT = 0
WRITER_SLOT = 1
# Common-RT layout: the io addresses first, then the core_row_start table. The table offset is handed to every kernel
# as the named CT arg `table_arg` (single source: these constants).
READER_ADDRESS_ARGS = 3  # common RT args 0..2: q, k, v addresses
WRITER_ADDRESS_ARGS = 4  # common RT args 0..3: output, q, v, k addresses (Q / V / a K part under split_reads)
COMPUTE_ADDRESS_ARGS = 0  # compute reads only the table

# ---------------------------------------------------------------------------
# K/V pair multicast (Refinement 2; op_design.md § Regimes "row-split + pair multicast")
# ---------------------------------------------------------------------------
# Reader per-core RT args: kv_depth role words (role of segment k, k < kv_depth), then per segment k < kv_depth a sender
# block [num_rects, MAX_MCAST_RECTS x RECT_WORDS] (zeros unless the role is SEND), where a rectangle is (lo.x, lo.y, hi.x,
# hi.y, remote, loopback, mode) in virtual NoC coordinates — the field order of dataflow_kernel_lib::
# RectangleRuntimeArguments (the reader orders lo / hi for its NoC). Semaphores: 2 a K/V ring slot (K data-ready,
# V data-ready), ids KV_SEM_BASE + 2*slot + {0, 1}.
ROLE_READ, ROLE_SEND, ROLE_RECEIVE = 0, 1, 2
MAX_MCAST_RECTS = 3  # a consecutive row-major core run is at most head row + full rows + tail row
RECT_WORDS = 7
SENDER_WORDS = 1 + MAX_MCAST_RECTS * RECT_WORDS
MODE_LOCAL_COPY, MODE_EXCLUDE_SOURCE, MODE_INCLUDE_SOURCE = 1, 2, 3  # dataflow_kernel_lib::SenderMcastMode
KV_SEM_BASE = 0


def _run_rectangles(first, last):
    """Logical rectangles (x0, y0, x1, y1) covering the row-major core run [first, last] of the grid."""
    width = GRID[0]
    (y0, x0), (y1, x1) = divmod(first, width), divmod(last, width)
    if y0 == y1:
        return [(x0, y0, x1, y1)]
    rects = []
    if x0 > 0:  # head: a partial first row
        rects.append((x0, y0, width - 1, y0))
        y0 += 1
    full_end = y1 if x1 == width - 1 else y1 - 1
    if full_end >= y0:
        rects.append((0, y0, width - 1, full_end))
    if x1 < width - 1:  # tail: a partial last row
        rects.append((0, y1, x1, y1))
    return rects


def kv_multicast_topology(p, device):
    """Per core: (roles[kv_depth], sender words[kv_depth * SENDER_WORDS]). A pair's GROUP is every core holding it as a
    segment of index < kv_depth — a slot no earlier segment of that core used, so the pair can land there at any time
    and no handshake is needed. Segment k of a core holds pair p0 + k and the K/V rings start p0 % kv_depth phantom
    slots in, so every member lands the pair at ring slot p % kv_depth (the identical L1 address multicast needs).
    Groups are consecutive core runs. The sender is the run's FIRST core: the reader's NoC (NOC0) routes +x / +y, so
    every receiver lies downstream of it — a sender behind its receivers would wrap each multicast around the whole
    torus row (measured: 8x512 126 -> 175 us). That core may hold the pair as a later segment, so the reader pre-sends
    it at t = 0, before its own schedule: no multicast waits on any core's reads or compute."""
    roles = [[ROLE_READ] * p.kv_depth for _ in range(p.num_cores)]
    sender_words = [[0] * (p.kv_depth * SENDER_WORDS) for _ in range(p.num_cores)]
    if not p.kv_mcast:
        return roles, sender_words
    members = {}
    for core in range(p.num_cores):
        for k, pair in enumerate(p.core_pairs(core)[: p.kv_depth]):
            members.setdefault(pair, []).append((core, k))
    for pair, group in members.items():
        if len(group) < 2:
            continue
        cores = [core for core, _ in group]
        assert cores == list(range(cores[0], cores[-1] + 1)), f"pair {pair}: group {cores} is not a core run"
        sender, sender_k = group[0]
        roles[sender][sender_k] = ROLE_SEND
        for core, k in group[1:]:
            roles[core][k] = ROLE_RECEIVE
        sx, sy = sender % GRID[0], sender // GRID[0]
        rects = _run_rectangles(cores[0], cores[-1])
        words = [len(rects)]
        for x0, y0, x1, y1 in rects:
            lo = device.worker_core_from_logical_core(ttnn.CoreCoord(x0, y0))
            hi = device.worker_core_from_logical_core(ttnn.CoreCoord(x1, y1))
            includes = x0 <= sx <= x1 and y0 <= sy <= y1
            remote = (x1 - x0 + 1) * (y1 - y0 + 1) - includes
            mode = MODE_EXCLUDE_SOURCE if not includes else (MODE_INCLUDE_SOURCE if remote else MODE_LOCAL_COPY)
            words += [lo.x, lo.y, hi.x, hi.y, remote, remote + 1, mode]
        sender_words[sender][sender_k * SENDER_WORDS : sender_k * SENDER_WORDS + len(words)] = words
    return roles, sender_words


# Debug scaffolding (ablation profiling, /perf-measure): NOMIC_SDPA_ABLATE="COMPUTE,READS" compiles every kernel
# with -DABLATE_COMPUTE -DABLATE_READS. Payloads are stubbed (outputs wrong), CB sync is kept. The per-stage device
# zones are permanent since Perf 1 (MaybeDeviceZoneScope; perf_tools/zone_report.py). Row-group schedule only: "EXP" stubs the
# pack-thread exp, "RELU" the per-window pack-ReLU toggles (outputs wrong). Never set in production.
def _ablation_defines():
    names = [n.strip() for n in os.environ.get("NOMIC_SDPA_ABLATE", "").split(",") if n.strip()]
    return [(f"ABLATE_{n}", "1") for n in names]


def circular_buffer_specs(p):
    """(index, pages, page bytes) of every CB: the single sizing source for the descriptor and
    circular_buffer_bytes() (l1_ledger.md § Total per-core footprint)."""
    q, n, hd, pv = p.q_block_rows, p.key_block_tiles, p.head_dim_tiles, p.pv_out_tiles
    # Row-group schedule (Refinement 3): the per-group buffers hold G rows. Scores and P are 2 groups deep (group t's
    # scores drain window by window while group t+1's fill; group t-1's P drains into P V while group t's fills), the
    # row stats one group (drained before the next group refills it); [O | l] never leaves DEST. The block-sequential
    # schedule holds the whole block, P in place in the scores.
    g = p.row_group_rows if p.exp_overlap else q
    specs = [
        (CB_QUERY, q * hd * p.q_depth, BF16_TILE_BYTES),
        (CB_KEY_T, n * hd * p.kv_depth, BF16_TILE_BYTES),
        (CB_VALUE_AUG, n * pv * p.kv_depth, BF16_TILE_BYTES),
        (CB_SCORES, (2 * g if p.exp_overlap else q) * n, BF16_TILE_BYTES),
        (CB_ROW_STATS, g, BF16_TILE_BYTES),
        (CB_OUT, q * hd * p.out_depth, BF16_TILE_BYTES),
        (CB_REDUCE_SCALER, 1, BF16_TILE_BYTES),
    ]
    if p.exp_overlap:
        specs.append((CB_PROBS, 2 * g * n, BF16_TILE_BYTES))
    return specs


def _cb(cores, index, pages, page_bytes):
    return ttnn.CBDescriptor(
        total_size=pages * page_bytes,
        core_ranges=cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=page_bytes)
        ],
    )


def _block_named_args(p):
    """Block knobs shared by all three kernels (each is a single CT arg, derived from the Plan)."""
    return [
        ("q_block_rows", p.q_block_rows),
        ("key_block_tiles", p.key_block_tiles),
        ("head_dim_tiles", p.head_dim_tiles),
        ("pv_out_tiles", p.pv_out_tiles),
        ("seq_tiles", p.seq_tiles),
        ("heads", HEADS),
        ("grid_x", GRID[0]),
        ("num_cores", p.num_cores),
        ("kv_depth", p.kv_depth),
        ("k_split_rows", p.k_split_rows),
        ("read_chunk", p.read_chunk),
    ]


def bind_call(descriptor, query, key, value, output):
    """Rebind a cached descriptor to this call's tensors: only the buffer addresses change between calls of a key."""
    kernels = descriptor.kernels
    reader_common = kernels[READER_SLOT].common_runtime_args
    addresses = (query, key, value)
    assert len(addresses) == READER_ADDRESS_ARGS
    for position, tensor in enumerate(addresses):
        reader_common[position] = tensor.buffer_address()
    writer_common = kernels[WRITER_SLOT].common_runtime_args
    writer_common[0] = output.buffer_address()
    writer_common[1] = query.buffer_address()
    writer_common[2] = value.buffer_address()
    writer_common[3] = key.buffer_address()


def create_program_descriptor(p, query, key, value, output, compute_config, scale):
    cores = p.cores
    cbs = [_cb(cores, index, pages, page_bytes) for index, pages, page_bytes in circular_buffer_specs(p)]
    table = list(p.core_row_start)
    block_args = _block_named_args(p)

    # ---------------- reader: K (K^T order), V (+ ones column), Q blocks ----------------
    reader_ct = []
    for tensor in (query, key, value):
        reader_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "nomic_sdpa_reader.cpp"),
        core_ranges=cores,
        compile_time_args=reader_ct,
        named_compile_time_args=[
            ("cb_query", CB_QUERY),
            ("cb_key_t", CB_KEY_T),
            ("cb_value_aug", CB_VALUE_AUG),
            ("q_depth", p.q_depth),
            ("q_lead_rows", p.q_lead_rows),
            ("kv_bank_stagger", int(bool(_op.KV_BANK_STAGGER))),
            ("table_arg", READER_ADDRESS_ARGS),
            ("kv_sem_base", KV_SEM_BASE),
            ("max_mcast_rects", MAX_MCAST_RECTS),
            ("split_reads", int(p.split_reads)),
            *block_args,
        ],
        defines=_ablation_defines(),
        common_runtime_args=[query.buffer_address(), key.buffer_address(), value.buffer_address(), *table],
        config=ttnn.ReaderConfigDescriptor(),
    )
    roles, sender_words = kv_multicast_topology(p, query.device())
    reader_rt = ttnn.RuntimeArgs()
    for core, coord in enumerate(p.core_list):
        reader_rt[coord.x][coord.y] = [*roles[core], *sender_words[core]]
    reader.runtime_args = reader_rt

    # ---------------- writer: reduce scaler, concatenated-heads output pages; Q and V reads under split_reads ----
    writer_ct = []
    for tensor in (output, query, value, key):
        writer_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "nomic_sdpa_writer.cpp"),
        core_ranges=cores,
        compile_time_args=writer_ct,
        named_compile_time_args=[
            ("cb_out", CB_OUT),
            ("cb_reduce_scaler", CB_REDUCE_SCALER),
            ("cb_query", CB_QUERY),
            ("cb_value_aug", CB_VALUE_AUG),
            ("cb_key_t", CB_KEY_T),
            ("q_depth", p.q_depth),
            ("q_lead_rows", p.q_lead_rows),
            ("kv_sem_base", KV_SEM_BASE),
            ("split_reads", int(p.split_reads)),
            ("table_arg", WRITER_ADDRESS_ARGS),
            *block_args,
        ],
        defines=_ablation_defines(),
        common_runtime_args=[
            output.buffer_address(),
            query.buffer_address(),
            value.buffer_address(),
            key.buffer_address(),
            *table,
        ],
        config=ttnn.WriterConfigDescriptor(),
    )
    writer_rt = ttnn.RuntimeArgs()  # the K/V multicast roles: V receives / sender hand-off under split_reads
    for core, coord in enumerate(p.core_list):
        writer_rt[coord.x][coord.y] = list(roles[core])
    writer.runtime_args = writer_rt

    # ---------------- compute ----------------
    config = ttnn.ComputeConfigDescriptor()
    config.math_fidelity = compute_config.math_fidelity
    config.fp32_dest_acc_en = bool(compute_config.fp32_dest_acc_en)
    config.math_approx_mode = bool(compute_config.math_approx_mode)
    config.dst_full_sync_en = bool(compute_config.dst_full_sync_en)
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "nomic_sdpa_compute.cpp"),
        core_ranges=cores,
        compile_time_args=[],
        named_compile_time_args=[
            ("cb_query", CB_QUERY),
            ("cb_key_t", CB_KEY_T),
            ("cb_value_aug", CB_VALUE_AUG),
            ("cb_scores", CB_SCORES),
            ("cb_row_stats", CB_ROW_STATS),
            ("cb_out", CB_OUT),
            ("cb_reduce_scaler", CB_REDUCE_SCALER),
            ("cb_probs", CB_PROBS),
            ("exp_overlap", int(p.exp_overlap)),
            ("row_group_rows", p.row_group_rows),
            ("apply_scale", 0 if float(scale) == 1.0 else 1),
            ("scale_bits", _scale_bits(scale)),
            ("exp_approx", 1 if config.math_approx_mode else 0),
            ("exp_form", _op.EXP_FORM),
            ("table_arg", COMPUTE_ADDRESS_ARGS),
            *block_args,
        ],
        defines=_ablation_defines(),
        common_runtime_args=table,
        config=config,
    )

    # 2 a K/V ring slot (K / V multicast data-ready) + 1 a slot (the reader -> writer V hand-off) + 2 split-startup
    # flags (writer -> reader: its segment-0 K part / its V half of a SEND slot are in L1)
    semaphores = [
        ttnn.SemaphoreDescriptor(id=KV_SEM_BASE + i, core_ranges=cores, initial_value=0)
        for i in range(3 * p.kv_depth + 2)
    ]
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=semaphores, cbs=cbs)
