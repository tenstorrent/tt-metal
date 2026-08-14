# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kernel program structure: does folding the reader + writer into a single compute kernel
help, or do the separate dataflow kernels earn their keep?

This is a GENERAL program-structure question, not a property of any one op. The concept: a
Tensix core has independent RISCs — dataflow readers/writers run on NCRISC/BRISC while compute
runs on TRISC — so a reader that arms a CB and a writer that drains one execute CONCURRENTLY
with the compute. Folding that arm/drain into the compute kernel serializes it onto the compute
thread. The tradeoff is the same for any compute op; the only thing the payload changes is how
much compute there is to hide the fixed arm/drain cost behind.

The payload here is incidental — a plain tilize (row-major -> tiled) on a same-spec resident L1
shard — chosen precisely because it strips everything else away: because the input/output shard
specs are IDENTICAL, both circular buffers alias directly onto the resident L1 shard buffers
(`cb_descriptor_from_sharded_tensor`), so there is NO DRAM and NO NoC traffic to muddy the
measurement. What is left is pure program structure around a bit of compute (`tilize_block`),
which is exactly what this example isolates. Any op with a reader/compute/writer split would
show the same effect.

Two ways to structure that program, measured against each other:

  * reader_compute_writer (BASELINE) — three kernels: a dataflow READER (on NCRISC) that arms
    the resident input CB (reserve+push, no NoC), the COMPUTE kernel (TRISC) that tilizes, and a
    dataflow WRITER (on BRISC) that drains the output CB (wait+pop, no NoC).

  * compute_only (CANDIDATE) — a single compute kernel that self-arms the resident input CB and
    self-drains the output CB, doing the whole conversion itself. No reader, no writer.

Same tilize, same CBs, same aliased buffers — the ONLY difference is program structure.

MEASURED RESULT (see report.md): folding is SLOWER, not faster — reader_compute_writer beats
compute_only across the board (~0.74x at 2 tiles/core, closing to ~0.95x at 64 tiles/core; same
direction per-launch and steady-state). The lesson is the opposite of the tempting intuition
that "fewer kernels = less overhead": the reader and writer run on their OWN RISC cores
(NCRISC / BRISC), so the CB arm and drain execute CONCURRENTLY with the tilize on the compute
core. Folding them into the compute kernel SERIALIZES the arm+drain onto the compute thread,
adding a fixed ~75-130 ns/launch that no NoC work is there to hide. The three "kernels" are not
overhead — they are three parallel RISC workers. The gap shrinks as tiles/core grows because
the tilize itself starts to dominate the fixed serialized arm/drain cost.

Correctness is the only pass/fail; DEVICE KERNEL DURATION [ns] is measured and reported.
"""

import ttnn

TILE = 32
CB_IN = 0  # row-major input shard (aliased); page overridden to one tile
CB_OUT = 16  # tiled output shard (aliased)

VARIANTS = ("reader_compute_writer", "compute_only")
BASELINE = "reader_compute_writer"

# ---------------------------------------------------------------------------
# Kernels (inline source). Compile-time args only — no runtime args needed since
# every core does the identical resident-shard conversion.
# ---------------------------------------------------------------------------

# Compute: tilize `num_rows` tile-rows (each Wt tiles wide) of the resident RM shard into the
# resident TILE shard. Arm/drain granularity is WHOLE-shard-once-per-iter (not per tile-row): the
# resident bytes are all present, so one reserve+push arms the whole input and one wait+pop drains
# the whole output. `fold` (=1 for compute_only) makes this kernel do that arm+drain itself, so no
# reader/writer are needed. `fold=0` (reader_compute_writer) leaves the arm to the reader and the
# drain to the writer. Either way the per-tile-row tilize is identical — the ONLY difference is
# whether the arm+drain live in this kernel (1 kernel) or in two separate dataflow kernels (3).
_COMPUTE_KERNEL = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t num_rows = get_compile_time_arg_val(3);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(4);
    constexpr uint32_t fold = get_compile_time_arg_val(5);
    constexpr uint32_t shard_tiles = num_rows * Wt;

    compute_kernel_hw_startup(cb_in, cb_out);
    tilize_init(cb_in, Wt, cb_out);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        if constexpr (fold) {
            // No reader: arm the whole resident RM input shard in one shot.
            cb_reserve_back(cb_in, shard_tiles);
            cb_push_back(cb_in, shard_tiles);
        }
        for (uint32_t r = 0; r < num_rows; ++r) {
            cb_wait_front(cb_in, Wt);
            cb_reserve_back(cb_out, Wt);
            tilize_block(cb_in, Wt, cb_out);
            cb_push_back(cb_out, Wt);
            cb_pop_front(cb_in, Wt);
        }
        if constexpr (fold) {
            // No writer: retire the whole output shard in one shot (bytes are already resident).
            cb_wait_front(cb_out, shard_tiles);
            cb_pop_front(cb_out, shard_tiles);
        }
    }

    tilize_uninit(cb_in, cb_out);
}
"""

# Dataflow READER (reader_compute_writer only): arm the whole resident input shard, no NoC.
_READER_KERNEL = r"""
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t num_rows = get_compile_time_arg_val(2);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(3);
    constexpr uint32_t shard_tiles = num_rows * Wt;
    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        cb_reserve_back(cb_in, shard_tiles);
        cb_push_back(cb_in, shard_tiles);
    }
}
"""

# Dataflow WRITER (reader_compute_writer only): drain the whole resident output shard, no NoC.
_WRITER_KERNEL = r"""
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t num_rows = get_compile_time_arg_val(2);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(3);
    constexpr uint32_t shard_tiles = num_rows * Wt;
    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        cb_wait_front(cb_out, shard_tiles);
        cb_pop_front(cb_out, shard_tiles);
    }
}
"""


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------
def _core_grid(ncores):
    """A 1-D row of `ncores` cores starting at (0,0)."""
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ncores - 1, 0))])


def sharded_memory_config(H, W, ncores):
    """HEIGHT-sharded, ROW_MAJOR orientation: each of `ncores` cores holds an [H/ncores, W] shard.

    With an explicit CoreRangeSet the binding requires `use_height_and_width_as_shard_shape=True`
    and `shape` = the per-core SHARD shape (not the full tensor shape)."""
    return ttnn.create_sharded_memory_config(
        shape=(H // ncores, W),
        core_grid=_core_grid(ncores),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _shard_tile_dims(H, W, ncores):
    """(num_rows per core = shard tile-rows, Wt = tile-columns). Requires tile-aligned shard."""
    shard_h = H // ncores
    if H % ncores or shard_h % TILE or W % TILE:
        raise ValueError(f"H/ncores and W must be tile-aligned: H={H} W={W} ncores={ncores}")
    return shard_h // TILE, W // TILE


def create_program_descriptor(input_tensor, output_tensor, *, variant, ncores, kernel_iters=1):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if input_tensor.dtype != ttnn.bfloat16 or input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("input must be bfloat16 ROW_MAJOR_LAYOUT")
    if output_tensor.dtype != ttnn.bfloat16 or output_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("output must be bfloat16 TILE_LAYOUT")

    H, W = int(input_tensor.shape[-2]), int(input_tensor.shape[-1])
    num_rows, Wt = _shard_tile_dims(H, W, ncores)
    grid = _core_grid(ncores)
    in_tile_size = ttnn.tile_size(ttnn.bfloat16)

    # Input CB aliased onto the resident RM shard; override its page size to one whole tile so the
    # tilize accounts in tiles while the row-major bytes stay put (Wt contiguous tile-pages == one
    # 32-row tile-row of the shard). Output CB aliased onto the resident TILE shard (already paged).
    cb_in = ttnn.cb_descriptor_from_sharded_tensor(CB_IN, input_tensor, core_ranges=grid)
    in_fds = cb_in.format_descriptors
    in_fds[0].page_size = in_tile_size
    cb_in.format_descriptors = in_fds
    cb_out = ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor, core_ranges=grid)

    fold = 1 if variant == "compute_only" else 0
    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=grid,
        compile_time_args=[CB_IN, CB_OUT, Wt, num_rows, kernel_iters, fold],
        config=ttnn.ComputeConfigDescriptor(),
    )
    if variant == "compute_only":
        return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=[cb_in, cb_out])

    reader = ttnn.KernelDescriptor(
        kernel_source=_READER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=grid,
        compile_time_args=[CB_IN, Wt, num_rows, kernel_iters],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=_WRITER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=grid,
        compile_time_args=[CB_OUT, Wt, num_rows, kernel_iters],
        config=ttnn.WriterConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=[cb_in, cb_out])


def run_op(input_tensor, *, variant, ncores, kernel_iters=1):
    H, W = int(input_tensor.shape[-2]), int(input_tensor.shape[-1])
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([H, W]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        sharded_memory_config(H, W, ncores),
    )
    descriptor = create_program_descriptor(
        input_tensor, output, variant=variant, ncores=ncores, kernel_iters=kernel_iters
    )
    return ttnn.generic_op([input_tensor, output], descriptor)
