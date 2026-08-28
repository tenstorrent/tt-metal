# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for toy_variance on a WIDTH-SHARDED input -- the cross-core case.

Width-sharding splits W, which is the axis the variance reduces over. That makes the reduced axis a
*dependent* axis split across cores, so the combine has to cross cores twice:

    round 1   each core reduces its own shard   -> partial contribution to the mean
              gather to root, root adds         -> the mean
              root broadcasts the mean back     -> every core needs it for round 2
    round 2   each core computes (x - mean)^2 over its shard -> partial contribution to the variance
              gather to root, root adds         -> the variance (then sqrt, for std_dev)

Both gathers are the `reduce_root_mcast` shape from
`ttnn/ttnn/operations/examples/tensix_all_reduce`: every core unicasts its block to one root, which
does the whole reduction. That example measures several other on-chip collective shapes against it
-- reduce-scatter (splitting *which* tiles each core reduces) and a tree gather (splitting *how* the
contributions travel) among them -- and on a wider group they beat a single root. This op stays on
the plain gather-to-root because its fan-in is one grid row and each contribution is Ht tiles, so
there is very little reduction work to spread; if the shard grid grows, that example is where to
look first.

Scaling detail that keeps the root's job to a plain add: every core reduces with scaler = 1/N over
the FULL row width, not its own slice, so each core's partial is already its share of the mean and
the combine never has to re-weight anything.

Placement is deliberately narrow -- see `validate()`. A shape that does not fit raises
NotImplementedError rather than silently falling back to the interleaved path.
"""

from pathlib import Path

import ttnn
from ttnn.mcast_spec import McastFamily


KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32

DFB_IN_SHARD = "in_shard"
DFB_SCALER = "scaler"
DFB_CENTERED_SQ = "centered_sq"
DFB_PARTIAL = "partial"
DFB_GATHER_MEAN = "gather_mean"
DFB_GATHER_VAR = "gather_var"
DFB_MEAN_SRC = "mean_src"
DFB_MEAN = "mean"
DFB_OUT = "out_tiles"

K_READER = "reader"
K_COMPUTE = "compute"
K_WRITER = "writer"

# Internal to this module -- see the note in toy_variance_program_artifacts.py.
TP_IN = "in"
TP_OUT = "out"

SEM_MEAN_ARRIVED = "mean_arrived"
SEM_VAR_ARRIVED = "var_arrived"

MCAST_PREFIX = "mean_bcast"


def _shard_row(input_tensor: ttnn.Tensor):
    """(cores, P) for a width-sharded input, or raise if the placement is not the supported one."""
    shard_spec = input_tensor.memory_config().shard_spec
    if shard_spec is None:
        raise NotImplementedError("toy_variance: sharded input has no shard_spec")
    cores = list(ttnn.corerange_to_cores(shard_spec.grid, None, True))
    return cores, len(cores)


def validate(input_tensor: ttnn.Tensor) -> None:
    """Gate the cross-core path. Everything here is a *narrowing* choice, not a hardware limit.

    The point of the toy is the cross-core combine, so the shape space is cut back until nothing
    else is in the way: no partial tiles, no ragged split, no second grid row.
    """
    memory_config = input_tensor.memory_config()
    if memory_config.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        raise NotImplementedError(
            f"toy_variance: sharded input must be WIDTH_SHARDED (got {memory_config.memory_layout}); "
            "the reduction is over W, so only a W split exercises the cross-core combine."
        )
    if memory_config.buffer_type != ttnn.BufferType.L1:
        raise NotImplementedError("toy_variance: width-sharded input must live in L1")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise NotImplementedError("toy_variance: the width-sharded path is TILE_LAYOUT only")
    if input_tensor.dtype != ttnn.bfloat16:
        raise NotImplementedError(f"toy_variance: the width-sharded path is bfloat16 only (got {input_tensor.dtype})")

    shape = list(input_tensor.shape)
    NC = 1
    for d in shape[:-2]:
        NC *= d
    if NC != 1:
        raise NotImplementedError(f"toy_variance: only NC=1 is supported (got NC={NC})")

    origin_H, origin_W = shape[-2], shape[-1]
    if origin_H % TILE_DIM != 0 or origin_W % TILE_DIM != 0:
        raise NotImplementedError(
            f"toy_variance: the width-sharded path requires tile-aligned H and W "
            f"(got H={origin_H}, W={origin_W}). Aligned W keeps the scaler a single full tile, so "
            "the cross-core combine is a plain add with no partial-scaler bookkeeping."
        )

    cores, num_cores = _shard_row(input_tensor)
    if num_cores < 2:
        raise NotImplementedError("toy_variance: the width-sharded path needs at least 2 cores to combine across")
    if any(core.y != cores[0].y for core in cores):
        raise NotImplementedError(
            "toy_variance: the width-sharded path expects the shard grid to be a single grid row "
            "(the gather fan-in and the mean broadcast are both along one row)."
        )

    Wt = origin_W // TILE_DIM
    if Wt % num_cores != 0:
        raise NotImplementedError(
            f"toy_variance: Wt={Wt} must divide evenly across {num_cores} cores; a ragged split would "
            "give the cores different tile counts, which is a work-distribution concern this toy "
            "deliberately keeps out of the way of the combine."
        )


def create_program_artifacts(input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor, *, std_dev: bool = False):
    shape = list(input_tensor.shape)
    origin_H, origin_W = shape[-2], shape[-1]
    Ht = origin_H // TILE_DIM
    Wt = origin_W // TILE_DIM

    cores, num_cores = _shard_row(input_tensor)
    Wt_local = Wt // num_cores
    shard_tiles = Ht * Wt_local

    device = input_tensor.device()
    grid = input_tensor.memory_config().shard_spec.grid
    root = cores[0]
    root_virtual = device.worker_core_from_logical_core(root)

    tile_bytes = ttnn.tile_size(ttnn.bfloat16)
    output_page_size = output_tensor.buffer_page_size()

    dfbs = [
        # The resident shard itself -- zero copy. Nothing reads it into L1 because it is already
        # there; the reader only credits it (see the reader kernel's note on the producer rule).
        ttnn.dfb_spec_from_sharded_tensor(DFB_IN_SHARD, input_tensor, borrowed_from=TP_IN),
        ttnn.DataflowBufferSpec(unique_id=DFB_SCALER, entry_size=tile_bytes, num_entries=1, data_format=ttnn.bfloat16),
        # (x - mean)^2 for the whole local shard: sub writes the block, then square consumes it, so
        # the buffer has to hold a full shard's worth rather than a streaming window.
        ttnn.DataflowBufferSpec(
            unique_id=DFB_CENTERED_SQ,
            entry_size=tile_bytes,
            num_entries=shard_tiles,
            data_format=input_tensor.dtype,
        ),
        # This core's contribution for the current round. Used once per round; 2*Ht so round 2 can
        # be staged while the writer is still draining round 1.
        ttnn.DataflowBufferSpec(
            unique_id=DFB_PARTIAL, entry_size=output_page_size, num_entries=2 * Ht, data_format=output_tensor.dtype
        ),
        # The two gather buffers. One per round rather than one reused: each is reserved once,
        # written once and pushed once, so `get_write_ptr()` is the base on every core with no
        # dependence on the ring pointers staying in step across cores (finding #9's hazard).
        ttnn.DataflowBufferSpec(
            unique_id=DFB_GATHER_MEAN,
            entry_size=output_page_size,
            num_entries=num_cores * Ht,
            data_format=output_tensor.dtype,
        ),
        ttnn.DataflowBufferSpec(
            unique_id=DFB_GATHER_VAR,
            entry_size=output_page_size,
            num_entries=num_cores * Ht,
            data_format=output_tensor.dtype,
        ),
        # The root's combined mean, handed to the reader to broadcast.
        ttnn.DataflowBufferSpec(
            unique_id=DFB_MEAN_SRC, entry_size=output_page_size, num_entries=Ht, data_format=output_tensor.dtype
        ),
        # The broadcast mean, as every core sees it.
        ttnn.DataflowBufferSpec(
            unique_id=DFB_MEAN, entry_size=output_page_size, num_entries=Ht, data_format=output_tensor.dtype
        ),
        ttnn.DataflowBufferSpec(
            unique_id=DFB_OUT, entry_size=output_page_size, num_entries=Ht, data_format=output_tensor.dtype
        ),
    ]

    shape_args = {"Ht": Ht, "Wt_local": Wt_local, "num_cores": num_cores, "shard_tiles": shard_tiles}

    reader = ttnn.KernelSpec(
        unique_id=K_READER,
        source=str(KERNEL_DIR / "sharded_reader.cpp"),
        hw_config=ttnn.create_reader_dm_config(),
        dfb_bindings=[
            ttnn.producer_of(DFB_IN_SHARD, DFB_IN_SHARD),
            ttnn.consumer_of(DFB_MEAN_SRC, DFB_MEAN_SRC),
            ttnn.producer_of(DFB_MEAN, DFB_MEAN),
        ],
        compile_time_args={**shape_args},
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["is_root"]),
    )

    compute = ttnn.KernelSpec(
        unique_id=K_COMPUTE,
        source=str(KERNEL_DIR / "sharded_compute.cpp"),
        hw_config=ttnn.ComputeGen1Config(),
        dfb_bindings=[
            ttnn.consumer_of(DFB_IN_SHARD, DFB_IN_SHARD),
            # reduce<> builds and reuses the scaler tile itself (ReduceScaler::compute_managed).
            ttnn.producer_of(DFB_SCALER, DFB_SCALER),
            ttnn.consumer_of(DFB_SCALER, DFB_SCALER),
            ttnn.producer_of(DFB_CENTERED_SQ, DFB_CENTERED_SQ),
            ttnn.consumer_of(DFB_CENTERED_SQ, DFB_CENTERED_SQ),
            ttnn.producer_of(DFB_PARTIAL, DFB_PARTIAL),
            ttnn.consumer_of(DFB_GATHER_MEAN, DFB_GATHER_MEAN),
            ttnn.consumer_of(DFB_GATHER_VAR, DFB_GATHER_VAR),
            ttnn.producer_of(DFB_MEAN_SRC, DFB_MEAN_SRC),
            ttnn.consumer_of(DFB_MEAN, DFB_MEAN),
            ttnn.producer_of(DFB_OUT, DFB_OUT),
        ],
        compile_time_args={**shape_args, "compute_std_dev": int(std_dev), "reduce_n": origin_W},
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["is_root"]),
    )

    writer = ttnn.KernelSpec(
        unique_id=K_WRITER,
        source=str(KERNEL_DIR / "sharded_writer.cpp"),
        hw_config=ttnn.create_writer_dm_config(),
        dfb_bindings=[
            ttnn.consumer_of(DFB_PARTIAL, DFB_PARTIAL),
            ttnn.producer_of(DFB_GATHER_MEAN, DFB_GATHER_MEAN),
            ttnn.producer_of(DFB_GATHER_VAR, DFB_GATHER_VAR),
            ttnn.consumer_of(DFB_OUT, DFB_OUT),
        ],
        semaphore_bindings=[
            ttnn.SemaphoreBinding(SEM_MEAN_ARRIVED, SEM_MEAN_ARRIVED),
            ttnn.SemaphoreBinding(SEM_VAR_ARRIVED, SEM_VAR_ARRIVED),
        ],
        tensor_bindings=[ttnn.TensorBinding(TP_OUT, TP_OUT)],
        compile_time_args={**shape_args, "root_x": root_virtual.x, "root_y": root_virtual.y},
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["is_root", "gather_slot"]),
    )

    spec = ttnn.ProgramSpec(
        name="toy_variance_width_sharded",
        kernels=[reader, compute, writer],
        dataflow_buffers=dfbs,
        semaphores=[
            ttnn.SemaphoreSpec(unique_id=SEM_MEAN_ARRIVED, target_nodes=grid),
            ttnn.SemaphoreSpec(unique_id=SEM_VAR_ARRIVED, target_nodes=grid),
        ],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id=TP_IN, spec=input_tensor.spec),
            ttnn.TensorParameter(unique_id=TP_OUT, spec=output_tensor.spec),
        ],
        work_units=[ttnn.WorkUnitSpec(name="main", kernels=[K_READER, K_COMPUTE, K_WRITER], target_nodes=grid)],
    )

    is_root_by_core = {core: int(core == root) for core in cores}
    run_args = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(kernel=K_READER, runtime_arg_values={"is_root": is_root_by_core}),
            ttnn.KernelRunArgs(kernel=K_COMPUTE, runtime_arg_values={"is_root": is_root_by_core}),
            ttnn.KernelRunArgs(
                kernel=K_WRITER,
                runtime_arg_values={
                    "is_root": is_root_by_core,
                    "gather_slot": {core: index for index, core in enumerate(cores)},
                },
            ),
        ]
    )

    # The mean broadcast: root -> the whole shard row. sender_index=0 makes cores[0] the sender,
    # which is the same core the gathers reduce onto.
    mcast = McastFamily(
        device,
        grid,
        MCAST_PREFIX,
        shape=ttnn.Mcast1DShape.PerRow,
        sender_index=0,
        config=ttnn.McastConfig(noc=ttnn.NOC.NOC_0),
    )
    mcast.attach(spec, run_args, kernels=[K_READER], cores=cores)

    tensor_indices = {TP_IN: 0, TP_OUT: 1}
    return spec, run_args, tensor_indices
