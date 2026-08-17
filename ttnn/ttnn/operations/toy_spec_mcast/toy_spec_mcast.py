# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Row-broadcast via a Metal 2.0 ProgramSpec + a kernel_lib mcast family.

`in` holds one tile per grid row. The sender core of each row reads its tile and multicasts it
across the row; every core writes what it received to its own output tile. So the output is `in`
replicated across the grid width, which a broken mcast cannot fake.

The point of the op is the mcast plumbing: one McastFamily.attach() call writes the semaphores,
bindings, named CT args and per-core varargs into the spec, and the kernel reads them back with
MCAST_ARGS(row). Neither side spells a CT or RT offset.
"""

from pathlib import Path

import ttnn
from ttnn.mcast_spec import McastFamily

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = 32

DFB_TILE = "tile"
K_READER = "reader"
K_WRITER = "writer"
TP_IN = "in"
TP_OUT = "out"
MCAST_PREFIX = "row"


def _grid(device, rows: int, cols: int):
    grid_size = device.compute_with_storage_grid_size()
    if rows > grid_size.y or cols > grid_size.x:
        raise NotImplementedError(f"grid {cols}x{rows} exceeds device grid {grid_size.x}x{grid_size.y}")
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))])


def toy_spec_mcast(inp: ttnn.Tensor, rows: int, cols: int) -> ttnn.Tensor:
    """`inp` is (1, 1, 32*rows, 32); returns (1, 1, 32*rows, 32*cols)."""
    if inp.layout != ttnn.TILE_LAYOUT or inp.dtype != ttnn.bfloat16:
        raise NotImplementedError("toy_spec_mcast requires TILE_LAYOUT bfloat16")
    if tuple(inp.shape) != (1, 1, TILE * rows, TILE):
        raise NotImplementedError(f"expected shape (1, 1, {TILE * rows}, {TILE}), got {tuple(inp.shape)}")

    device = inp.device()
    grid = _grid(device, rows, cols)

    out_spec = ttnn.TensorSpec(
        ttnn.Shape([1, 1, TILE * rows, TILE * cols]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.TensorMemoryLayout.INTERLEAVED,
    )
    out = ttnn.allocate_tensor_on_device(out_spec, device)

    tile_bytes = inp.buffer_page_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(rows) for x in range(cols)]

    mcast = McastFamily(
        device,
        grid,
        MCAST_PREFIX,
        shape=ttnn.Mcast1DShape.PerRow,
        sender_index=0,
        config=ttnn.McastConfig(noc=ttnn.NOC.NOC_0),
    )

    spec = ttnn.ProgramSpec(
        name="toy_spec_mcast",
        kernels=[
            ttnn.KernelSpec(
                unique_id=K_READER,
                source=str(KERNEL_DIR / "reader.cpp"),
                hw_config=ttnn.create_reader_dm_config(),
                dfb_bindings=[ttnn.producer_of(DFB_TILE, DFB_TILE)],
                tensor_bindings=[ttnn.TensorBinding(TP_IN, TP_IN)],
                runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["row_page", "is_sender"]),
            ),
            ttnn.KernelSpec(
                unique_id=K_WRITER,
                source=str(KERNEL_DIR / "writer.cpp"),
                hw_config=ttnn.create_writer_dm_config(),
                dfb_bindings=[ttnn.consumer_of(DFB_TILE, DFB_TILE)],
                tensor_bindings=[ttnn.TensorBinding(TP_OUT, TP_OUT)],
                runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["out_page"]),
            ),
        ],
        dataflow_buffers=[
            ttnn.DataflowBufferSpec(unique_id=DFB_TILE, entry_size=tile_bytes, num_entries=1, data_format=ttnn.bfloat16)
        ],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id=TP_IN, spec=inp.spec),
            ttnn.TensorParameter(unique_id=TP_OUT, spec=out.spec),
        ],
        work_units=[ttnn.WorkUnitSpec(name="main", kernels=[K_READER, K_WRITER], target_nodes=grid)],
    )

    run_args = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(
                kernel=K_READER,
                runtime_arg_values={
                    "row_page": {c: c.y for c in cores},
                    "is_sender": {c: int(mcast.is_sender(c)) for c in cores},
                },
            ),
            ttnn.KernelRunArgs(
                kernel=K_WRITER,
                runtime_arg_values={"out_page": {c: c.y * cols + c.x for c in cores}},
            ),
        ]
    )

    mcast.attach(spec, run_args, kernels=[K_READER], cores=cores)

    return ttnn.generic_op([inp, out], spec, run_args, {TP_IN: 0, TP_OUT: 1})
