# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for nomic_moe_combine (op_design.md § Circular Buffers / § Work Distribution).

Every CB size, loop count and the grid derive from the Plan (`_plan()` in the op module): shard_row_tiles (r),
shard_col_tiles (W), rows_depth_tile_rows, the per-core (tr0, core_row_tiles, g, bias copy). Nothing is sized from a
whole-op dimension.
"""

import importlib
import os
from pathlib import Path

import ttnn

# Tunable knobs are read from the op module at call time (single source; a sweep can patch them). import_module, not
# `from . import`: the package re-exports the function of the same name, which shadows the submodule attribute.
_op = importlib.import_module(".nomic_moe_combine", __package__)
from .nomic_moe_combine import BF16_TILE_BYTES, FP32_TILE_BYTES, INDEX_TILE_ROW_BYTES, TOP_K

KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices (semantic names; op_design.md § Circular Buffers).
CB_ROWS_CHOICE0 = 0
CB_ROWS_CHOICE1 = 1
CB_WEIGHTS_RAW_CHOICE0 = 2
CB_WEIGHTS_RAW_CHOICE1 = 3
CB_WEIGHTS_BCAST = 4
CB_BIAS_TILES = 5
CB_INDEX_CHOICE0 = 6
CB_INDEX_CHOICE1 = 7
CB_ROWS_TILIZED = 8  # realization R2 only
CB_OUTPUT_SHARD = 16

# Slots of the per-call state in the cached descriptor (bind_call): the readers are kernels[0 .. TOP_K), the output
# shard's CB is cbs[0].
READER_KERNEL_SLOTS = tuple(range(TOP_K))
OUTPUT_CB_SLOT = 0

# Semaphore of choice k (group_shared_index_weights): on a peer, the count of items its group leader delivered.
DELIVERED_SEMAPHORE = (0, 1)

CB_ROWS = (CB_ROWS_CHOICE0, CB_ROWS_CHOICE1)
CB_WEIGHTS_RAW = (CB_WEIGHTS_RAW_CHOICE0, CB_WEIGHTS_RAW_CHOICE1)
CB_INDEX = (CB_INDEX_CHOICE0, CB_INDEX_CHOICE1)


# Debug scaffolding (ablation profiling, /perf-measure): NOMIC_MOE_COMBINE_ABLATE="A,B" compiles the kernels with
# -DABLATE_A -DABLATE_B. Outputs are wrong under any ablation; never set in production.
def _ablation_defines():
    names = [n.strip() for n in os.environ.get("NOMIC_MOE_COMBINE_ABLATE", "").split(",") if n.strip()]
    return [(f"ABLATE_{n}", "1") for n in names]


def _cb(core_ranges, pages, index, dtype, page_size):
    return ttnn.CBDescriptor(
        total_size=pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def circular_buffer_specs(r, w, rows_depth_tile_rows, fused_tilize):
    """(index, pages, dtype, page bytes) of every CB the op allocates beyond the output shard (which is backed by the
    output tensor). The single source of the CB sizing: the descriptor allocates exactly these, and
    circular_buffer_bytes() / validate()'s budget check sum them (l1_ledger.md § Total)."""
    specs = []
    for k in range(TOP_K):
        specs.append((CB_ROWS[k], rows_depth_tile_rows * w, ttnn.bfloat16, BF16_TILE_BYTES))
        specs.append((CB_WEIGHTS_RAW[k], r, ttnn.float32, FP32_TILE_BYTES))
        specs.append((CB_INDEX[k], r, ttnn.uint32, INDEX_TILE_ROW_BYTES))
    if fused_tilize:
        specs.append((CB_WEIGHTS_BCAST, TOP_K * r, ttnn.float32, FP32_TILE_BYTES))  # R1
    else:
        specs.append((CB_ROWS_TILIZED, TOP_K * w, ttnn.bfloat16, BF16_TILE_BYTES))  # R2
    specs.append((CB_BIAS_TILES, w, ttnn.float32, FP32_TILE_BYTES))
    return specs


def _input_addresses(expert_rows, combine_index, combine_weights, bias_rows):
    """The readers' common runtime args: the four input buffer addresses (the only per-call reader args)."""
    return [
        expert_rows.buffer_address(),
        combine_index.buffer_address(),
        combine_weights.buffer_address(),
        bias_rows.buffer_address(),
    ]


def bind_call(descriptor, plan, expert_rows, combine_index, combine_weights, bias_rows, output):
    """Rebind a (cached) descriptor to this call's tensors: the readers' address args (common args 0 .. 3, patched in
    place) and the output shard's CB. Everything else in the descriptor depends only on the call-state key
    (nomic_moe_combine._CALL_CACHE)."""
    addresses = _input_addresses(expert_rows, combine_index, combine_weights, bias_rows)
    kernels = descriptor.kernels  # references into the descriptor
    for slot in READER_KERNEL_SLOTS:
        common = kernels[slot].common_runtime_args  # reference: element writes land in the descriptor
        for position, address in enumerate(addresses):
            common[position] = address
    output_cb = ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT_SHARD, output, core_ranges=plan.cores)
    descriptor.cbs[OUTPUT_CB_SLOT].set_buffer_from_cb(output_cb)


def create_program_descriptor(plan, expert_rows, combine_index, combine_weights, bias_rows, output, compute_config):
    """The whole program for one call-state key, bound to these tensors. Built once per key and cached by the op;
    later calls only bind_call() it."""
    cores = plan.cores
    r = plan.shard_row_tiles
    w = plan.shard_col_tiles

    # ---------------- circular buffers ----------------
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT_SHARD, output, core_ranges=cores)]  # OUTPUT_CB_SLOT
    for index, pages, dtype, page_size in circular_buffer_specs(r, w, plan.rows_depth_tile_rows, _op.FUSED_TILIZE):
        cbs.append(_cb(cores, pages, index, dtype, page_size))

    # ---------------- core tables (common runtime args; no per-core runtime args) ----------------
    # generic_op re-sends every per-core runtime arg on each call (~0.3 us a core a kernel of host time), so the
    # per-core values live in tables every core gets once: core i (= y * grid_x + x, _plan checks it) reads record i.
    # Readers: [addresses (4), records (N), noc_xy (N)]; compute: [records (N)].
    addresses = _input_addresses(expert_rows, combine_index, combine_weights, bias_rows)
    records = [_op.pack_core_record(*values) for values in plan.per_core]
    device = output.device()
    noc_xy = []
    for core in plan.core_list:
        noc = device.worker_core_from_logical_core(core)
        noc_xy.append(noc.x | noc.y << 16)
    reader_common = [*addresses, *records, *noc_xy]

    # ---------------- readers (one per choice) ----------------
    accessor_ct = []
    accessor_ct.extend(ttnn.TensorAccessorArgs(combine_index).get_compile_time_args())
    accessor_ct.extend(ttnn.TensorAccessorArgs(combine_weights).get_compile_time_args())
    accessor_ct.extend(ttnn.TensorAccessorArgs(bias_rows).get_compile_time_args())
    kernels = []
    for k in READER_KERNEL_SLOTS:  # kernels[k] is choice k's reader
        reader_ct = [
            k,  # choice
            CB_ROWS[k],
            CB_WEIGHTS_RAW[k],
            CB_BIAS_TILES,
            CB_INDEX[k],
            w,  # shard_col_tiles
            r,  # shard_row_tiles
            plan.rows_depth_tile_rows,
            expert_rows.buffer_aligned_page_size(),  # bank stride of one expert row
            _op.BATCH_READS,
            _op.BATCH_DEPTH,
            plan.token_tiles,  # T' / 32: weight tiles per choice
            int(_op.INDEX_FIRST),
            r if _op.WEIGHTS_LOOKAHEAD_ROWS is None else max(1, min(r, _op.WEIGHTS_LOOKAHEAD_ROWS)),
            _op.SIDE_READ_VC,
            int(_op.WEIGHTS_ONE_READ),
            _op.SIDE_READS_AFTER_BATCHES,
            plan.hidden_split,  # G
            int(_op.SHARE_INDEX),
            int(_op.SHARE_WEIGHTS),
            DELIVERED_SEMAPHORE[k],
            _op.leader_member(k, plan.hidden_split),
            _op.GRID[0],  # grid_x: core i = y * grid_x + x
            len(plan.core_list),  # num_cores: the records' and noc_xy tables' length
            _op.CORE_RECORD_FIELD_BITS,
            *accessor_ct,
        ]
        processor, noc = _op.READER_PLACEMENT[k]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "nomic_moe_combine_reader.cpp"),
                core_ranges=cores,
                compile_time_args=reader_ct,
                defines=_ablation_defines(),
                common_runtime_args=reader_common,
                config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
            )
        )

    # ---------------- compute ----------------
    compute_ct = [
        w,
        r,
        CB_ROWS_CHOICE0,
        CB_ROWS_CHOICE1,
        CB_WEIGHTS_RAW_CHOICE0,
        CB_WEIGHTS_RAW_CHOICE1,
        CB_WEIGHTS_BCAST,
        CB_BIAS_TILES,
        CB_OUTPUT_SHARD,
        CB_ROWS_TILIZED,
        int(_op.FUSED_TILIZE),
        _op.DEST_GROUP_TILES,
        TOP_K,
        int(_op.BIAS_SEED_SRCB),
        _op.GRID[0],  # grid_x
        _op.CORE_RECORD_FIELD_BITS,
    ]
    kernels.append(
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "nomic_moe_combine_compute.cpp"),
            core_ranges=cores,
            compile_time_args=compute_ct,
            defines=_ablation_defines(),
            common_runtime_args=records,
            config=compute_config,
        )
    )

    semaphores = [
        ttnn.SemaphoreDescriptor(id=DELIVERED_SEMAPHORE[k], core_ranges=cores, initial_value=0) for k in range(TOP_K)
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
