# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Two single-tensor Metal 2.0 ProgramSpec programs, both dispatched with a ONE-element io_tensors.

- `toy_spec_fill`: write-only generator. No kernel reads a tensor; the value is materialized in L1
  and NOC-written out. Its only tensor is the output.
- `toy_spec_square_`: true in-place. One TensorParameter is bound by the reader (read) and by the
  writer (write), so the same tensor is the input and the output.

`ttnn.generic_op` requires only that io_tensors' LAST element is the output tensor, which both
satisfy with a single element.
"""

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

DFB_A = "in_a"
DFB_B = "in_b"
DFB_OUT = "out"
K_FILL = "fill"
K_READER = "reader"
K_COMPUTE = "compute"
K_WRITER = "writer"
TP_OUT = "out"  # fill program: the generated output
TP_T = "t"  # in-place program: read and written


def bf16_bits(value: float) -> int:
    """The bf16 encoding of `value` = the top 16 bits of its fp32 encoding.

    Truncating rather than rounding, so callers should use exactly representable values.
    """
    return int.from_bytes(struct.pack(">f", value)[:2], "big")


def _split_tiles(num_tiles: int, grid_size) -> list[tuple]:
    """(core, num_tiles, start_id) per participating core, row-major over the grid."""
    cores = [ttnn.CoreCoord(x, y) for y in range(grid_size.y) for x in range(grid_size.x)]
    cores = cores[: min(len(cores), num_tiles)]

    base, rem = divmod(num_tiles, len(cores))
    assignment = []
    start = 0
    for i, core in enumerate(cores):
        count = base + (1 if i < rem else 0)
        assignment.append((core, count, start))
        start += count
    return assignment


def _plan(tensor: ttnn.Tensor):
    num_tiles = tensor.buffer_num_pages()
    tile_bytes = tensor.buffer_page_size()
    assignment = _split_tiles(num_tiles, tensor.device().compute_with_storage_grid_size())
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core, _, _ in assignment])
    num_tiles_by_core = {core: count for core, count, _ in assignment}
    start_id_by_core = {core: start for core, _, start in assignment}
    return tile_bytes, core_set, num_tiles_by_core, start_id_by_core


def _writer_kernel(tensor_parameter: str) -> ttnn.KernelSpec:
    return ttnn.KernelSpec(
        unique_id=K_WRITER,
        source=str(KERNEL_DIR / "writer.cpp"),
        hw_config=ttnn.create_writer_dm_config(),
        dfb_bindings=[ttnn.consumer_of(DFB_OUT, "out")],
        tensor_bindings=[ttnn.TensorBinding(tensor_parameter, "out")],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles", "start_id"]),
    )


def create_fill_spec(out: ttnn.Tensor, value: float):
    """Write-only program: `out` is the only tensor, and no kernel reads a tensor."""
    tile_bytes, core_set, num_tiles_by_core, start_id_by_core = _plan(out)

    fill = ttnn.KernelSpec(
        unique_id=K_FILL,
        source=str(KERNEL_DIR / "fill.cpp"),
        hw_config=ttnn.create_reader_dm_config(),
        dfb_bindings=[ttnn.producer_of(DFB_OUT, "out")],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles", "fill_bits"]),
    )

    spec = ttnn.ProgramSpec(
        name="toy_spec_fill",
        kernels=[fill, _writer_kernel(TP_OUT)],
        dataflow_buffers=[
            ttnn.DataflowBufferSpec(unique_id=DFB_OUT, entry_size=tile_bytes, num_entries=2, data_format=out.dtype)
        ],
        tensor_parameters=[ttnn.TensorParameter(unique_id=TP_OUT, spec=out.spec)],
        work_units=[ttnn.WorkUnitSpec(name="main", kernels=[K_FILL, K_WRITER], target_nodes=core_set)],
    )

    fill_bits_by_core = {core: bf16_bits(value) for core in num_tiles_by_core}
    run_args = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(
                kernel=K_FILL,
                runtime_arg_values={"num_tiles": num_tiles_by_core, "fill_bits": fill_bits_by_core},
            ),
            ttnn.KernelRunArgs(
                kernel=K_WRITER,
                runtime_arg_values={"num_tiles": num_tiles_by_core, "start_id": start_id_by_core},
            ),
        ]
    )
    return spec, run_args


def create_square_spec(t: ttnn.Tensor):
    """In-place program: one TensorParameter, read by the reader and written by the writer."""
    tile_bytes, core_set, num_tiles_by_core, start_id_by_core = _plan(t)

    reader = ttnn.KernelSpec(
        unique_id=K_READER,
        source=str(KERNEL_DIR / "reader.cpp"),
        hw_config=ttnn.create_reader_dm_config(),
        dfb_bindings=[ttnn.producer_of(DFB_A, "in_a"), ttnn.producer_of(DFB_B, "in_b")],
        tensor_bindings=[ttnn.TensorBinding(TP_T, "t")],
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

    spec = ttnn.ProgramSpec(
        name="toy_spec_square_inplace",
        kernels=[reader, compute, _writer_kernel(TP_T)],
        dataflow_buffers=[
            ttnn.DataflowBufferSpec(unique_id=name, entry_size=tile_bytes, num_entries=2, data_format=t.dtype)
            for name in (DFB_A, DFB_B, DFB_OUT)
        ],
        tensor_parameters=[ttnn.TensorParameter(unique_id=TP_T, spec=t.spec)],
        work_units=[ttnn.WorkUnitSpec(name="main", kernels=[K_READER, K_COMPUTE, K_WRITER], target_nodes=core_set)],
    )

    dm_args = {"num_tiles": num_tiles_by_core, "start_id": start_id_by_core}
    run_args = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(kernel=K_READER, runtime_arg_values=dm_args),
            ttnn.KernelRunArgs(kernel=K_COMPUTE, runtime_arg_values={"num_tiles": num_tiles_by_core}),
            ttnn.KernelRunArgs(kernel=K_WRITER, runtime_arg_values=dm_args),
        ]
    )
    return spec, run_args


def _check(t: ttnn.Tensor) -> None:
    if t.layout != ttnn.TILE_LAYOUT or t.dtype != ttnn.bfloat16:
        raise NotImplementedError(f"requires TILE_LAYOUT bfloat16, got {t.layout} and {t.dtype}")


def toy_spec_fill(out: ttnn.Tensor, value: float) -> ttnn.Tensor:
    """Fill `out` with `value`. Write-only: io_tensors is just [out]."""
    _check(out)
    spec, run_args = create_fill_spec(out, value)
    return ttnn.generic_op([out], spec, run_args, {TP_OUT: 0})


def toy_spec_square_(t: ttnn.Tensor) -> ttnn.Tensor:
    """Square `t` in place. io_tensors is just [t]: it is both the input and the output."""
    _check(t)
    spec, run_args = create_square_spec(t)
    return ttnn.generic_op([t], spec, run_args, {TP_T: 0})
