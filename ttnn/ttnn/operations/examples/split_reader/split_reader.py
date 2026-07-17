# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Dual-RISC-V split-reader example for a small-transaction row gather-copy.

Each core in the top row owns a height-sharded L1 slice. One consumer core below
the row streams the tiles in their original order, deliberately reconstructing
each 2 KiB tile with a configurable number of NoC reads. Its BRISC writes the
same tensor to DRAM, so the operation stays entirely on that core's
data-movement processors.

``split_reader=False`` puts every read transaction on the normal reader
(RISCV_1, NoC0), while BRISC writes the gathered tiles to DRAM. With
``split_reader=True``, BRISC still writes the results but also gathers the
second half of every block on NoC1. The input, block boundaries, DRAM writes,
and output are otherwise identical.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
TILE_BYTES = TILE * TILE * 2
DEFAULT_TRANSACTION_BYTES = TILE * 2  # 64 B: 32 independently issued transactions per tile

CB_GATHER_0 = 0
CB_GATHER_1 = 1

SPLIT_READER_MODES = (False, True)


def _row_cores(num_cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))])


def _single_core(x, y):
    core = ttnn.CoreCoord(x, y)
    return ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])


def make_row_sharded_memory_config(shape, num_cores):
    """Height-shard ``shape`` over logical cores ``(0,0)..(N-1,0)`` in L1."""
    h, w = list(shape)
    if h % num_cores:
        raise ValueError(f"split_reader example: height {h} must divide evenly over {num_cores} cores")
    return ttnn.create_sharded_memory_config(
        [h // num_cores, w],
        core_grid=_row_cores(num_cores),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _validate(input_tensor, num_cores, block_tiles, transaction_bytes):
    shape = list(input_tensor.shape)
    if len(shape) != 2:
        raise ValueError(f"split_reader example: rank must be 2, got {len(shape)}")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("split_reader example: input must be TILE_LAYOUT")
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError("split_reader example: input must be bfloat16")
    if num_cores < 2:
        raise ValueError(f"split_reader example: num_cores must be >= 2, got {num_cores}")
    if block_tiles < 4 or block_tiles % 2:
        raise ValueError(f"split_reader example: block_tiles must be even and >= 4, got {block_tiles}")
    if transaction_bytes < 16 or transaction_bytes > TILE_BYTES or transaction_bytes % 16:
        raise ValueError(
            f"split_reader example: transaction_bytes must be a multiple of 16 in [16, {TILE_BYTES}], "
            f"got {transaction_bytes}"
        )
    if TILE_BYTES % transaction_bytes:
        raise ValueError(
            f"split_reader example: transaction_bytes={transaction_bytes} must divide tile size {TILE_BYTES}"
        )

    grid = input_tensor.device().compute_with_storage_grid_size()
    if num_cores > grid.x:
        raise ValueError(
            f"split_reader example: num_cores={num_cores} does not fit in the {grid.x}-core-wide worker row"
        )
    if grid.y < 2:
        raise ValueError("split_reader example: a consumer core below the source row is required")

    h, w = shape
    if w != TILE or h % (num_cores * TILE):
        raise ValueError(
            "split_reader example: expected shape [num_cores * tiles_per_core * 32, 32], "
            f"got {shape} for num_cores={num_cores}"
        )
    total_tiles = h // TILE
    if total_tiles % block_tiles:
        raise ValueError(f"split_reader example: total tiles {total_tiles} must divide into block_tiles={block_tiles}")
    if total_tiles // block_tiles < 2:
        raise ValueError("split_reader example: the input must contain at least two streaming blocks")

    memory_config = input_tensor.memory_config()
    expected_grid = _row_cores(num_cores)
    if not memory_config.is_sharded() or memory_config.memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        raise ValueError("split_reader example: input must be height-sharded in L1")
    if memory_config.buffer_type != ttnn.BufferType.L1:
        raise ValueError("split_reader example: input shards must be in L1")
    if memory_config.shard_spec.grid != expected_grid:
        raise ValueError("split_reader example: input shard grid must be the top source row (0,0)..(N-1,0)")
    expected_shard_shape = [h // num_cores, w]
    if list(memory_config.shard_spec.shape) != expected_shard_shape:
        raise ValueError(
            f"split_reader example: expected per-core shard shape {expected_shard_shape}, "
            f"got {list(memory_config.shard_spec.shape)}"
        )


def _create_program_descriptor(input_tensor, output_tensor, *, split_reader, num_cores, block_tiles, transaction_bytes):
    device = input_tensor.device()
    consumer_core = _single_core(0, 1)
    tiles_per_source = list(input_tensor.shape)[0] // (num_cores * TILE)
    total_tiles = num_cores * tiles_per_source
    num_blocks = total_tiles // block_tiles
    half_block_tiles = block_tiles // 2

    # Two blocks deep permits reader/writer overlap. Both modes use the same
    # CBs: the baseline's one reader fills both, while split mode assigns one CB
    # (one half of every block) to each data-movement RISC-V.
    gather_cbs = [
        ttnn.CBDescriptor(
            total_size=2 * half_block_tiles * TILE_BYTES,
            core_ranges=consumer_core,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=ttnn.bfloat16, page_size=TILE_BYTES)
            ],
        )
        for cb_id in (CB_GATHER_0, CB_GATHER_1)
    ]
    # Tensor shards have one common L1 base address. Readers receive the virtual
    # NoC coordinates for each logical source core and select the remote shard by
    # coordinate, exactly as sharded transpose/all-gather kernels do.
    physical_cores = [device.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)) for x in range(num_cores)]
    noc_x = [core.x for core in physical_cores]
    noc_y = [core.y for core in physical_cores]

    def make_reader(num_halves):
        compile_args = [
            CB_GATHER_0,
            CB_GATHER_1,
            tiles_per_source,
            TILE_BYTES,
            transaction_bytes,
            num_cores,
            block_tiles,
            num_blocks,
            0,
            num_halves,
        ]
        runtime_args = ttnn.RuntimeArgs()
        runtime_args[0][1] = [input_tensor.buffer_address(), *noc_x, *noc_y]
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gather_reader.cpp"),
            core_ranges=consumer_core,
            compile_time_args=compile_args,
            runtime_args=runtime_args,
            config=ttnn.ReaderConfigDescriptor(),
        )

    reader_kernel = make_reader(1 if split_reader else 2)

    writer_compile_args = [
        CB_GATHER_0,
        CB_GATHER_1,
        tiles_per_source,
        TILE_BYTES,
        transaction_bytes,
        num_cores,
        block_tiles,
        num_blocks,
        int(split_reader),
    ]
    writer_compile_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_runtime_args = ttnn.RuntimeArgs()
    writer_runtime_args[0][1] = [
        output_tensor.buffer_address(),
        input_tensor.buffer_address(),
        *noc_x,
        *noc_y,
    ]
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "gather_writer.cpp"),
        core_ranges=consumer_core,
        compile_time_args=writer_compile_args,
        runtime_args=writer_runtime_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel],
        semaphores=[],
        cbs=gather_cbs,
    )


def row_gather_copy(
    input_tensor: ttnn.Tensor,
    *,
    split_reader: bool = True,
    num_cores: int = 8,
    block_tiles: int = 8,
    transaction_bytes: int = DEFAULT_TRANSACTION_BYTES,
):
    """Gather top-row L1 shards into an identical DRAM tensor.

    One consumer core receives the row's sequence and preserves its tile order.
    Split mode additionally gives BRISC the second-half reads in every streaming
    block; the DRAM writes are identical in both modes and no Tensix compute
    kernel is used.

    Args:
        split_reader: False has NCRISC fill both block halves while BRISC only
            writes output. True has NCRISC fill the first half and BRISC gather
            the second half before writing that block to DRAM.
        num_cores: Number of source shards in the top logical row.
        block_tiles: Even number of input tiles consumed per streaming iteration;
            must be at least four and divide the total gathered tile count.
        transaction_bytes: Bytes issued by each NoC read. Must be a multiple of
            16 and divide a 2 KiB tile. Smaller values issue more transactions.
    """
    if split_reader not in SPLIT_READER_MODES:
        raise ValueError(f"split_reader example: split_reader must be bool, got {split_reader!r}")
    _validate(input_tensor, num_cores, block_tiles, transaction_bytes)

    device = input_tensor.device()
    total_tiles = list(input_tensor.shape)[0] // TILE
    output_shape = [total_tiles * TILE, TILE]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    descriptor = _create_program_descriptor(
        input_tensor,
        output_tensor,
        split_reader=split_reader,
        num_cores=num_cores,
        block_tiles=block_tiles,
        transaction_bytes=transaction_bytes,
    )
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
