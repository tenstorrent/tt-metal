# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 ProgramSpec for toy_spec_mul (elementwise multiply, tiled, interleaved)."""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

DFB_A = "in_a"
DFB_B = "in_b"
DFB_OUT = "out"
K_READER = "reader"
K_COMPUTE = "compute"
K_WRITER = "writer"
TP_A = "a"
TP_B = "b"
TP_OUT = "out"


def _split_tiles(num_tiles: int, grid_size, *, tile_limit: int | None = None) -> list[tuple]:
    """(core, num_tiles, start_id) per participating core, row-major over the grid.

    The core set is always derived from the full tile count, so `tile_limit` changes only
    the runtime arg VALUES and leaves the ProgramSpec byte-identical. Cores past the limit
    get zero tiles.
    """
    cores = [ttnn.CoreCoord(x, y) for y in range(grid_size.y) for x in range(grid_size.x)]
    cores = cores[: min(len(cores), num_tiles)]

    work = num_tiles if tile_limit is None else min(tile_limit, num_tiles)
    num_workers = min(len(cores), work) or 1
    base, rem = divmod(work, num_workers)

    assignment = []
    start = 0
    for i, core in enumerate(cores):
        count = (base + (1 if i < rem else 0)) if i < num_workers else 0
        assignment.append((core, count, start))
        start += count
    return assignment


def create_program_spec(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor, *, tile_limit: int | None = None):
    num_tiles = out.buffer_num_pages()
    tile_bytes = out.buffer_page_size()
    grid_size = a.device().compute_with_storage_grid_size()
    assignment = _split_tiles(num_tiles, grid_size, tile_limit=tile_limit)

    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core, _, _ in assignment])

    dfbs = [
        ttnn.DataflowBufferSpec(unique_id=name, entry_size=tile_bytes, num_entries=2, data_format=dtype)
        for name, dtype in ((DFB_A, a.dtype), (DFB_B, b.dtype), (DFB_OUT, out.dtype))
    ]

    tensor_parameters = [
        ttnn.TensorParameter(unique_id=TP_A, spec=a.spec),
        ttnn.TensorParameter(unique_id=TP_B, spec=b.spec),
        ttnn.TensorParameter(unique_id=TP_OUT, spec=out.spec),
    ]

    reader = ttnn.KernelSpec(
        unique_id=K_READER,
        source=str(KERNEL_DIR / "reader.cpp"),
        hw_config=ttnn.create_reader_dm_config(),
        dfb_bindings=[
            ttnn.producer_of(DFB_A, "in_a"),
            ttnn.producer_of(DFB_B, "in_b"),
        ],
        tensor_bindings=[
            ttnn.TensorBinding(TP_A, "a"),
            ttnn.TensorBinding(TP_B, "b"),
        ],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles", "start_id"]),
    )

    compute = ttnn.KernelSpec(
        unique_id=K_COMPUTE,
        source=str(KERNEL_DIR / "compute.cpp"),
        hw_config=ttnn.ComputeGen1Config(),
        dfb_bindings=[
            ttnn.consumer_of(DFB_A, "in_a"),
            ttnn.consumer_of(DFB_B, "in_b"),
            ttnn.producer_of(DFB_OUT, "out"),
        ],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )

    writer = ttnn.KernelSpec(
        unique_id=K_WRITER,
        source=str(KERNEL_DIR / "writer.cpp"),
        hw_config=ttnn.create_writer_dm_config(),
        dfb_bindings=[ttnn.consumer_of(DFB_OUT, "out")],
        tensor_bindings=[ttnn.TensorBinding(TP_OUT, "out")],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles", "start_id"]),
    )

    spec = ttnn.ProgramSpec(
        name="toy_spec_mul",
        kernels=[reader, compute, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=tensor_parameters,
        work_units=[ttnn.WorkUnitSpec(name="main", kernels=[K_READER, K_COMPUTE, K_WRITER], target_nodes=core_set)],
    )

    num_tiles_by_core = {core: count for core, count, _ in assignment}
    start_id_by_core = {core: start for core, _, start in assignment}
    dm_args = {"num_tiles": num_tiles_by_core, "start_id": start_id_by_core}

    run_args = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(kernel=K_READER, runtime_arg_values=dm_args),
            ttnn.KernelRunArgs(kernel=K_COMPUTE, runtime_arg_values={"num_tiles": num_tiles_by_core}),
            ttnn.KernelRunArgs(kernel=K_WRITER, runtime_arg_values=dm_args),
        ]
    )

    return spec, run_args
