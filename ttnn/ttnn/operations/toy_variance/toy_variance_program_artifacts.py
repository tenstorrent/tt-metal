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
DFB_ACCUMULATOR = "reduce_accumulator"

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

    BLOCK_SIZE = pick_block_size(Wt, block_size)
    NUM_BLOCKS = Wt // BLOCK_SIZE

    # Variance reduces over the *real* W (= origin_W). With scaler = 1/N built into the reduce, SUM
    # produces means directly. The partial scaler tile zeros out the contributions of the
    # padded positions in the last W-tile, so origin_W is the correct N.

    input_page_size = input_tensor.buffer_page_size()
    output_page_size = output_tensor.buffer_page_size()

    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    tiles_per_block = Ht * BLOCK_SIZE
    scaler_tile_bytes = ttnn.tile_size(ttnn.bfloat16)

    planner = ttnn.reduce_planner

    def reduce_spec(width, dtype):
        return ttnn.TensorSpec(
            ttnn.Shape([Ht * TILE_DIM, width]),
            dtype,
            ttnn.TILE_LAYOUT,
            ttnn.TensorMemoryLayout.INTERLEAVED,
            None,
            ttnn.BufferType.L1,
        )

    configs = []
    call_count = min(NUM_BLOCKS, 3)
    for i in range(call_count):
        width = origin_W - (NUM_BLOCKS - 1) * BLOCK_SIZE * TILE_DIM if i + 1 == call_count else BLOCK_SIZE * TILE_DIM
        configs.append(
            (
                0,
                planner.ReduceCallConfig(
                    input_spec=reduce_spec(width, input_tensor.dtype),
                    output_spec=reduce_spec(1, output_tensor.dtype),
                    reduce_math=planner.ReduceMath.SUM,
                    reduce_dim=planner.ReduceDimension.ROW,
                    scalar=1.0 / origin_W,
                    fp32_mode=planner.ReduceFp32Mode.FAST,
                    max_input_cb_bytes=BLOCK_SIZE * input_page_size,
                ),
            )
        )
    sequence = planner.make_reduce_sequence_plan(
        reductions=configs,
        cb_ids=planner.ReduceSequenceCbIds(auxiliary_cb_id=1, accumulator_cb_id=3, output_cb_id=2),
        hardware=planner.ReduceHardwareConfig(
            arch=input_tensor.device().arch(),
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
            available_l1_bytes=ttnn.get_max_worker_l1_unreserved_size(),
        ),
    )
    reduce_args = []
    sequence.append_to(reduce_args)
    auxiliary_args = []
    sequence.auxiliary.append_to(auxiliary_args)

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
        # The host recipe includes any full/tail scalars or masking tiles.
        ttnn.DataflowBufferSpec(
            unique_id=DFB_SCALER,
            entry_size=scaler_tile_bytes,
            num_entries=len(sequence.auxiliary.tiles),
            data_format=ttnn.bfloat16,
        ),
        ttnn.DataflowBufferSpec(
            unique_id=DFB_ACCUMULATOR,
            entry_size=output_page_size,
            num_entries=max(2 * Ht, 2),
            data_format=output_tensor.dtype,
        ),
        # mean: persistent across all of pass 2 (WaitUpfrontNoPop). After pass 1 holds Ht tiles;
        # capacity must be >= Ht.
        ttnn.DataflowBufferSpec(
            unique_id=DFB_MEAN,
            entry_size=output_page_size,
            num_entries=max(2 * Ht, 2),
            data_format=output_tensor.dtype,
        ),
        # variance: final output of pass 2; intermediate blocks use reduce_accumulator.
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
    }

    reader = ttnn.KernelSpec(
        unique_id=K_READER,
        source=str(KERNEL_DIR / "reader.cpp"),
        hw_config=ttnn.create_reader_dm_config(),
        dfb_bindings=[
            ttnn.producer_of(DFB_IN, DFB_IN),
            ttnn.producer_of(DFB_SCALER, DFB_SCALER),
        ],
        tensor_bindings=[ttnn.TensorBinding(TP_IN, TP_IN)],
        compile_time_args={
            **shape_args,
            "auxiliary_tiles": len(sequence.auxiliary.tiles),
        },
        advanced_options=ttnn.KernelAdvancedOptions(compile_time_varargs=auxiliary_args),
    )

    compute = ttnn.KernelSpec(
        unique_id=K_COMPUTE,
        source=str(KERNEL_DIR / "compute.cpp"),
        hw_config=ttnn.ComputeGen1Config(),
        dfb_bindings=[
            ttnn.consumer_of(DFB_IN, DFB_IN),
            ttnn.consumer_of(DFB_SCALER, DFB_SCALER),
            ttnn.producer_of(DFB_ACCUMULATOR, DFB_ACCUMULATOR),
            ttnn.consumer_of(DFB_ACCUMULATOR, DFB_ACCUMULATOR),
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
            "auxiliary_tiles": len(sequence.auxiliary.tiles),
        },
        advanced_options=ttnn.KernelAdvancedOptions(compile_time_varargs=reduce_args),
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
