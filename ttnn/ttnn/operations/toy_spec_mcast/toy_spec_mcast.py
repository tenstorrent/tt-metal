# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Broadcast via a Metal 2.0 ProgramSpec + a kernel_lib mcast family, in both mcast topologies.

`toy_spec_mcast` (1D): `in` holds one tile per grid row. The sender core of each row reads its tile
and multicasts it across the row; every core writes what it received to its own output tile. So the
output is `in` replicated across the grid width, which a broken mcast cannot fake.

`toy_spec_mcast_2d` (2D): `in` is ONE tile. A single sender core reads it and multicasts it over the
whole receiver rectangle in one shot; every participating core writes its copy to its own output
tile. The sender may sit inside the rectangle or outside it.

The point of the op is the mcast plumbing: one McastFamily.attach() call writes the semaphores,
bindings, named CT args and per-core varargs into the spec, and the kernel reads them back with
MCAST_ARGS(row). Neither side spells a CT or RT offset -- and the SAME reader kernel serves both
topologies unchanged, because the wire (five named CT words + a 6-word vararg block) is identical.
"""

from pathlib import Path

import ttnn
from ttnn.mcast_spec import McastFamily

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = 32

DFB_TILE = "tile"
K_READER = "reader"
K_WRITER = "writer"
# Internal to this module: each factory returns the tensor bindings it built, so these names
# never have to travel to the entry points.
TP_IN = "in"
TP_OUT = "out"
MCAST_PREFIX = "row"


def _grid(device, rows: int, cols: int):
    grid_size = device.compute_with_storage_grid_size()
    if rows > grid_size.y or cols > grid_size.x:
        raise NotImplementedError(f"grid {cols}x{rows} exceeds device grid {grid_size.x}x{grid_size.y}")
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))])


def create_program_artifacts(inp: ttnn.Tensor, rows: int, cols: int):
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

    # io_tensors is [inp, out].
    return out, spec, run_args, {TP_IN: 0, TP_OUT: 1}


def create_2d_program_artifacts(inp: ttnn.Tensor, rows: int, cols: int, sender=None):
    """One 2D mcast of a single tile over a `cols` x `rows` rectangle at the grid origin.

    `inp` is (1, 1, 32, 32). `sender` is the one broadcasting core, inside the rectangle or outside
    it; it defaults to the rectangle's origin. Returns (1, 1, 32, 32 * n) holding one copy of the
    tile per participating core, in the enumeration order of `McastFamily.nodes` -- a flat mapping
    rather than a grid-shaped one so that a sender outside the rectangle needs no special case.
    """
    if inp.layout != ttnn.TILE_LAYOUT or inp.dtype != ttnn.bfloat16:
        raise NotImplementedError("toy_spec_mcast_2d requires TILE_LAYOUT bfloat16")
    if tuple(inp.shape) != (1, 1, TILE, TILE):
        raise NotImplementedError(f"expected shape (1, 1, {TILE}, {TILE}), got {tuple(inp.shape)}")

    device = inp.device()
    rect = _grid(device, rows, cols)
    sender = ttnn.CoreCoord(0, 0) if sender is None else ttnn.CoreCoord(*sender)
    grid_size = device.compute_with_storage_grid_size()
    if sender.x >= grid_size.x or sender.y >= grid_size.y:
        raise NotImplementedError(f"sender ({sender.x},{sender.y}) is outside device grid {grid_size.x}x{grid_size.y}")

    # The kernel spells MCAST_ARGS(row), so the prefix stays "row" for the 2D family too: the macro
    # names a family, not a topology.
    mcast = McastFamily(device, rect, MCAST_PREFIX, sender=sender, config=ttnn.McastConfig(noc=ttnn.NOC.NOC_0))
    # nodes is the rectangle plus the sender when the sender sits outside it; every one of those
    # cores runs the program, so it is both the work unit and the output page map.
    cores = list(ttnn.corerange_to_cores(mcast.nodes, None, True))

    out_spec = ttnn.TensorSpec(
        ttnn.Shape([1, 1, TILE, TILE * len(cores)]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.TensorMemoryLayout.INTERLEAVED,
    )
    out = ttnn.allocate_tensor_on_device(out_spec, device)
    tile_bytes = inp.buffer_page_size()

    spec = ttnn.ProgramSpec(
        name="toy_spec_mcast_2d",
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
        work_units=[ttnn.WorkUnitSpec(name="main", kernels=[K_READER, K_WRITER], target_nodes=mcast.nodes)],
    )

    run_args = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(
                kernel=K_READER,
                runtime_arg_values={
                    "row_page": {c: 0 for c in cores},
                    "is_sender": {c: int(mcast.is_sender(c)) for c in cores},
                },
            ),
            ttnn.KernelRunArgs(
                kernel=K_WRITER,
                runtime_arg_values={"out_page": {c: i for i, c in enumerate(cores)}},
            ),
        ]
    )

    mcast.attach(spec, run_args, kernels=[K_READER], cores=cores)

    # io_tensors is [inp, out].
    return out, spec, run_args, {TP_IN: 0, TP_OUT: 1}


# ---------------------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------------------
# Both factories above return FOUR values, not the usual three. The output tensor's shape is
# derived from the mcast topology -- for the 2D program it is one tile per participating core,
# which is only known once McastFamily has resolved its node set -- so the factory allocates it
# rather than making the entry point rebuild the family just to learn the shape. Everything else
# follows the template: the factory owns every name it declares and hands back the tensor
# bindings, so nothing below names a TensorParameter.


def toy_spec_mcast(inp: ttnn.Tensor, rows: int, cols: int) -> ttnn.Tensor:
    """`inp` is (1, 1, 32*rows, 32); returns (1, 1, 32*rows, 32*cols)."""
    out, spec, run_args, tensor_indices = create_program_artifacts(inp, rows, cols)
    return ttnn.generic_op([inp, out], spec, run_args, tensor_indices)


def toy_spec_mcast_2d(inp: ttnn.Tensor, rows: int, cols: int, sender=None) -> ttnn.Tensor:
    """One 2D mcast of a single tile over a `cols` x `rows` rectangle at the grid origin."""
    out, spec, run_args, tensor_indices = create_2d_program_artifacts(inp, rows, cols, sender)
    return ttnn.generic_op([inp, out], spec, run_args, tensor_indices)
