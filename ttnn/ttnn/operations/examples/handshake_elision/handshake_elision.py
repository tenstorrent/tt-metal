# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Handshake-elision example: what the reader/compute/writer CB protocol costs when
the data is already resident, and what deleting it is actually worth.

The op is a tilize (row-major -> tiled) of a HEIGHT-sharded bfloat16 tensor whose
input and output shards have the same spec and live in the same core's L1. Both
circular buffers are aliased directly onto the shard buffers, so the reader has
nothing to fetch and the writer has nothing to send: not one byte crosses the NoC.
What is left is the compute (a `tilize_block` per tile-row) and the program
structure around it -- which is the thing under study.

ONE compile-time constant in the compute kernel, `no_handshake`, selects the arm:

  variant="handshake"     no_handshake=0   WITH synchronization (baseline)
      Three kernels. A reader (NCRISC) publishes the input CB once per iteration,
      a writer (BRISC) retires the output CB once per iteration, and the compute
      kernel runs the per-tile-row credit protocol -- cb_wait_front and
      cb_reserve_back before every tilize, cb_push_back and cb_pop_front after.
      This is what an op skeleton gives you.

  variant="no_handshake"  no_handshake=1   WITHOUT synchronization
      One kernel, no CB protocol at all. The whole input is present before launch
      and the whole output is consumed after it, so no wait can ever be needed and
      no credit is ever owed. Tile-rows are addressed by tile index from the CB
      base pointers, which never advance.

Everything else is identical across the arms: the same tilize call, the same two
aliased CBs, the same cores, the same shard. The arms differ only in whether the
CB counters are touched (and, as a consequence, in whether the two dataflow
kernels exist at all).

Why the "no_handshake" arm is legal, and when it is not: it rests on both CBs being
exactly one shard deep and the shard being fully resident on both sides. The
moment a dataflow kernel has to move bytes (a DRAM or remote-L1 source, an
output that must be written out) the compute kernel has something to WAIT for
and the protocol is load-bearing again. Delete the handshake only where it
provably guards nothing.

See README.md for the mechanism, the measured ns, and the CLI.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
CB_IN = 0  # row-major input shard (aliased); page size overridden to one tile
CB_OUT = 16  # tiled output shard (aliased)

# Baseline first: the three-kernel skeleton is what you write by default.
VARIANTS = ("handshake", "no_handshake")
# The compute kernel's `no_handshake` compile-time constant, per variant.
NO_HANDSHAKE = {"handshake": 0, "no_handshake": 1}
# Kernels launched per core, per variant -- the table column the result is read against.
KERNELS = {"handshake": 3, "no_handshake": 1}

DTYPE = ttnn.bfloat16


def _core_grid(num_cores, row=0):
    """`num_cores` cores in ONE row, x = 0..num_cores-1. Same for every variant."""
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, row), ttnn.CoreCoord(num_cores - 1, row))])


def sharded_memory_config(shard_ht, shard_wt, num_cores):
    """HEIGHT-sharded L1: each core holds one [shard_ht*32, shard_wt*32] shard."""
    return ttnn.create_sharded_memory_config(
        shape=(shard_ht * TILE, shard_wt * TILE),
        core_grid=_core_grid(num_cores),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def validate(input_tensor, shard_ht, shard_wt, num_cores):
    shape = list(input_tensor.shape)
    if len(shape) != 2:
        raise ValueError(f"handshake_elision example: rank must be 2, got {len(shape)}")
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT or input_tensor.dtype != DTYPE:
        raise ValueError("handshake_elision example: input must be bfloat16 ROW_MAJOR_LAYOUT")
    if not input_tensor.is_sharded():
        raise ValueError("handshake_elision example: input must be L1 height-sharded (see sharded_memory_config)")
    if shape != [shard_ht * TILE * num_cores, shard_wt * TILE]:
        raise ValueError(
            f"handshake_elision example: shape {shape} must be [{shard_ht}*32*{num_cores}, {shard_wt}*32] "
            "so every core holds exactly one tile-aligned shard"
        )
    grid = input_tensor.device().compute_with_storage_grid_size()
    if not (1 <= num_cores <= grid.x):
        raise ValueError(f"handshake_elision example: num_cores must be in [1, {grid.x}] (one row), got {num_cores}")


def _aliased_cb(cb_id, tensor, core_ranges):
    """A CB aliased onto the tensor's resident L1 shard, paged in whole tiles.

    A ROW_MAJOR shard's natural page is one row; the tilize accounts in tiles, so
    the page size is overridden to one tile. The bytes do not move: `Wt` tile-pages
    are exactly one 32-row band of the row-major shard.
    """
    cb = ttnn.cb_descriptor_from_sharded_tensor(cb_id, tensor, core_ranges=core_ranges)
    fds = cb.format_descriptors
    fds[0].page_size = ttnn.tile_size(DTYPE)
    cb.format_descriptors = fds
    return cb


def create_program_descriptor(input_tensor, output_tensor, *, variant, shard_ht, shard_wt, num_cores, kernel_iters):
    if variant not in VARIANTS:
        raise ValueError(f"handshake_elision example: variant must be one of {VARIANTS}, got {variant!r}")
    grid = _core_grid(num_cores)
    shard_tiles = shard_ht * shard_wt

    cb_in = _aliased_cb(CB_IN, input_tensor, grid)
    cb_out = _aliased_cb(CB_OUT, output_tensor, grid)

    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "he_compute.cpp"),
        core_ranges=grid,
        compile_time_args=[CB_IN, CB_OUT, shard_wt, shard_ht, kernel_iters, NO_HANDSHAKE[variant]],
        config=ttnn.ComputeConfigDescriptor(),
    )
    kernels = [compute]

    if variant == "handshake":
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "he_arm_reader.cpp"),
                core_ranges=grid,
                compile_time_args=[CB_IN, shard_tiles, kernel_iters],
                config=ttnn.ReaderConfigDescriptor(),  # NCRISC
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "he_drain_writer.cpp"),
                core_ranges=grid,
                compile_time_args=[CB_OUT, shard_tiles, kernel_iters],
                config=ttnn.WriterConfigDescriptor(),  # BRISC
            ),
            compute,
        ]
    assert len(kernels) == KERNELS[variant]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb_in, cb_out])


def handshake_elision(
    input_tensor: ttnn.Tensor,
    *,
    variant: str = "no_handshake",
    shard_ht: int,
    shard_wt: int,
    num_cores: int = 1,
    kernel_iters: int = 1,
) -> ttnn.Tensor:
    """Tilize a same-spec L1 height-sharded bf16 tensor in place, with a chosen handshake arm.

    Args:
        variant: "handshake" (baseline: reader + compute + writer with the
            per-tile-row CB protocol) or "no_handshake" (one kernel, no CB protocol
            -- legal because both CBs are exactly one resident shard deep and
            nothing moves through the NoC).
        shard_ht, shard_wt: the per-core shard in tiles; `shard_ht * shard_wt` is
            the work per core. The saving under study is a fixed per-launch cost, so
            sweep this down to a couple of tiles to see it, and up to see it fade.
        num_cores: cores in row 0 each holding one shard; the tensor must be
            [shard_ht*32*num_cores, shard_wt*32].
        kernel_iters: in-kernel repeat. 1 = per-launch latency (where a fixed cost
            shows); large = steady-state, which amortizes it.

    The output is the tiled layout of the input; values are bitwise identical for
    every arm.
    """
    if kernel_iters < 1:
        raise ValueError(f"handshake_elision example: kernel_iters must be >= 1, got {kernel_iters}")
    validate(input_tensor, shard_ht, shard_wt, num_cores)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        DTYPE,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        sharded_memory_config(shard_ht, shard_wt, num_cores),
    )
    descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        variant=variant,
        shard_ht=shard_ht,
        shard_wt=shard_wt,
        num_cores=num_cores,
        kernel_iters=kernel_iters,
    )
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
