# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for toy_variance on an INTERLEAVED input (single core).

Computes per-row population variance Var(x) = E[(x - E[x])^2] using a two-pass
streaming algorithm:
    Pass 1: stream x -> dfb mean     = E[x]              via accumulating reduce<>
    Pass 2: stream x -> dfb variance = E[(x-mean)^2]     via sub<COL> + square + accumulating reduce<>
The reduction axis is chunked into num_blocks blocks of block_size tiles each so W can be
arbitrarily wide (e.g. 32 x 64000) without exceeding L1.

Restricted to NC=1 (no leading batch tile rows beyond the H direction) since binary_op_helpers'
BinaryInputBlockShape carries no NC dimension.

Host model: ProgramSpec, not ProgramDescriptor. That choice is what makes the kernels' `dfb::`,
`args::` and `tensor::` names exist at all -- `is_metal2_kernel()` gates all three generated
surfaces together, and it is set only for ProgramSpec-created kernels. See
`.claude/PROGRAMSPEC_MIGRATION_PLAYBOOK.md` §1.
"""

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32

DFB_IN = "in_tiles"
DFB_CENTERED_SQ = "centered_sq"
DFB_SCALER = "scaler"
DFB_MEAN = "mean"
DFB_VARIANCE = "variance"
DFB_OUT = "out_tiles"

K_READER = "reader"
K_COMPUTE = "compute"
K_WRITER = "writer"

# Internal to this module: the factory returns the tensor bindings it built, so these
# names never have to travel to the entry point.
TP_IN = "in"
TP_OUT = "out"


def pick_block_size(Wt: int, requested: int | None) -> int:
    """Pick a block_size that divides Wt. Default to the largest divisor of Wt that is <= 8."""
    if requested is not None:
        if Wt % requested != 0:
            raise ValueError(f"toy_variance: block_size={requested} does not divide Wt={Wt}")
        return requested
    for candidate in range(min(8, Wt), 0, -1):
        if Wt % candidate == 0:
            return candidate
    return 1


def fp32_bits(value: float) -> int:
    return struct.unpack("I", struct.pack("f", value))[0]


def create_program_artifacts(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    block_size: int | None = None,
    std_dev: bool = False,
):
    input_shape = list(input_tensor.shape)
    origin_W = input_shape[-1]
    origin_H = input_shape[-2]

    NC = 1
    for d in input_shape[:-2]:
        NC *= d
    if NC != 1:
        raise ValueError(
            f"toy_variance: only NC=1 is supported (got leading dims producing NC={NC}); "
            "binary_op_helpers' BinaryInputBlockShape has no batch dimension."
        )

    # Tile counts cover the padded shape; the reduce uses a partial scaler on the last W-tile to
    # suppress contributions from padded positions. The H direction can also be padded -- those
    # padded rows produce garbage outputs that the caller is responsible for slicing off.
    Wt = (origin_W + TILE_DIM - 1) // TILE_DIM
    Ht = (origin_H + TILE_DIM - 1) // TILE_DIM

    partial_w = origin_W % TILE_DIM
    has_partial_w = partial_w != 0

    BLOCK_SIZE = pick_block_size(Wt, block_size)
    NUM_BLOCKS = Wt // BLOCK_SIZE

    # Variance reduces over the *real* W (= origin_W), so the compute kernel runs an AVG reduce with
    # reduce_factor = origin_W. The partial scaler tile zeros out the contributions of the
    # (TILE_DIM - partial_w) padded positions in the last W-tile, so origin_W is the correct N.

    input_page_size = input_tensor.buffer_page_size()
    output_page_size = output_tensor.buffer_page_size()

    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    tiles_per_block = Ht * BLOCK_SIZE
    scaler_tile_bytes = ttnn.tile_size(ttnn.bfloat16)

    dfbs = [
        # in_tiles: per-tile streaming for both passes. Double-buffer one block of work for
        # reader/compute pipelining.
        ttnn.DataflowBufferSpec(
            unique_id=DFB_IN,
            entry_size=input_page_size,
            num_entries=2 * tiles_per_block,
            data_format=input_tensor.dtype,
        ),
        # centered_sq: holds (x - mean)^2 tiles for one block. The sub helper (PerTile output)
        # pushes them sequentially; the streaming reduce pops them one at a time. Both are
        # sequential within compute, so a single block's worth of entries is sufficient; 2x for
        # headroom.
        ttnn.DataflowBufferSpec(
            unique_id=DFB_CENTERED_SQ,
            entry_size=input_page_size,
            num_entries=2 * tiles_per_block,
            data_format=input_tensor.dtype,
        ),
        # scaler: 2 tiles when has_partial_w (full + partial scaler), else 1.
        ttnn.DataflowBufferSpec(
            unique_id=DFB_SCALER,
            entry_size=scaler_tile_bytes,
            num_entries=2 if has_partial_w else 1,
            data_format=ttnn.bfloat16,
        ),
        # mean: persistent across all of pass 2 (WaitUpfrontNoPop). After pass 1 holds Ht tiles;
        # capacity must be >= Ht.
        ttnn.DataflowBufferSpec(
            unique_id=DFB_MEAN,
            entry_size=output_page_size,
            num_entries=max(2 * Ht, 2),
            data_format=output_tensor.dtype,
        ),
        # variance: streaming reduce accumulator for pass 2. Pop-1/push-1 per ht per block, so
        # capacity >= Ht. 2x for safety.
        ttnn.DataflowBufferSpec(
            unique_id=DFB_VARIANCE,
            entry_size=output_page_size,
            num_entries=max(2 * Ht, 2),
            data_format=output_tensor.dtype,
        ),
        ttnn.DataflowBufferSpec(
            unique_id=DFB_OUT,
            entry_size=output_page_size,
            num_entries=2,
            data_format=output_tensor.dtype,
        ),
    ]

    shape_args = {
        "Ht": Ht,
        "Wt": Wt,
        "block_size": BLOCK_SIZE,
        "num_blocks": NUM_BLOCKS,
        "has_partial_w": int(has_partial_w),
    }

    reader = ttnn.KernelSpec(
        unique_id=K_READER,
        source=str(KERNEL_DIR / "reader.cpp"),
        hw_config=ttnn.create_reader_dm_config(),
        dfb_bindings=[
            ttnn.producer_of(DFB_IN, DFB_IN),
        ],
        tensor_bindings=[ttnn.TensorBinding(TP_IN, TP_IN)],
        compile_time_args={**shape_args},
    )

    compute = ttnn.KernelSpec(
        unique_id=K_COMPUTE,
        source=str(KERNEL_DIR / "compute.cpp"),
        hw_config=ttnn.ComputeGen1Config(),
        dfb_bindings=[
            ttnn.consumer_of(DFB_IN, DFB_IN),
            # reduce<> builds and reuses the scaler tiles itself (ReduceScaler::compute_managed).
            ttnn.producer_of(DFB_SCALER, DFB_SCALER),
            ttnn.consumer_of(DFB_SCALER, DFB_SCALER),
            ttnn.producer_of(DFB_CENTERED_SQ, DFB_CENTERED_SQ),
            ttnn.consumer_of(DFB_CENTERED_SQ, DFB_CENTERED_SQ),
            ttnn.producer_of(DFB_MEAN, DFB_MEAN),
            ttnn.consumer_of(DFB_MEAN, DFB_MEAN),
            ttnn.producer_of(DFB_VARIANCE, DFB_VARIANCE),
            ttnn.consumer_of(DFB_VARIANCE, DFB_VARIANCE),
            ttnn.producer_of(DFB_OUT, DFB_OUT),
        ],
        compile_time_args={
            **shape_args,
            "compute_std_dev": int(std_dev),
            # Valid positions in the last W-tile; TILE_DIM when W is tile-aligned.
            "partial_w": partial_w if has_partial_w else TILE_DIM,
            # AVG reduces over the real W, so the managed scaler is 1/origin_W.
            "reduce_n": origin_W,
        },
    )

    writer = ttnn.KernelSpec(
        unique_id=K_WRITER,
        source=str(KERNEL_DIR / "writer.cpp"),
        hw_config=ttnn.create_writer_dm_config(),
        dfb_bindings=[ttnn.consumer_of(DFB_OUT, DFB_OUT)],
        tensor_bindings=[ttnn.TensorBinding(TP_OUT, TP_OUT)],
        compile_time_args={"num_tiles": output_tensor.buffer_num_pages()},
    )

    spec = ttnn.ProgramSpec(
        name="toy_variance",
        kernels=[reader, compute, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=[
            ttnn.TensorParameter(unique_id=TP_IN, spec=input_tensor.spec),
            ttnn.TensorParameter(unique_id=TP_OUT, spec=output_tensor.spec),
        ],
        work_units=[ttnn.WorkUnitSpec(name="main", kernels=[K_READER, K_COMPUTE, K_WRITER], target_nodes=core_grid)],
    )

    run_args = ttnn.ProgramRunArgs(kernel_run_args=[])

    # Bind each declared TensorParameter to its position in the caller's io_tensors list,
    # which is [input_tensor, output_tensor].
    tensor_indices = {TP_IN: 0, TP_OUT: 1}
    return spec, run_args, tensor_indices
