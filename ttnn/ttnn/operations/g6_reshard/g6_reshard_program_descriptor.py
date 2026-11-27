# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_reshard ProgramDescriptor: PULL regime (op_design.md, Blocking Model).

Every core that holds a block of out runs ISSUERS_PER_CORE instances of one dataflow kernel. Each instance
owns a contiguous range of the block's tile rows, walks the pieces of in0's blocks that overlap it on device,
and reads them straight into out's shard. No CB, no compute kernel, no semaphore.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
ISSUER_KERNEL = str(KERNEL_DIR / "g6_reshard_reader.cpp")

TILE_DIM = 32

# L1 the model leaves to the op on every core (BB-05).
L1_FREE_BYTES = 1347776

# (processor, NoC) of each issuer slot. Issuer k of a core runs on ISSUER_PLACEMENTS[k].
ISSUER_PLACEMENTS = (
    (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
    (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
)

# Knob: data-movement RISC-Vs that split one out block's tile rows (1 or 2).
ISSUERS_PER_CORE = 2

# Per-core RT args of an issuer, in kernel order.
RT_BLOCK_ROW_BEGIN, RT_ROW_BEGIN, RT_ROW_END, RT_COL_BEGIN, RT_COL_END = range(5)
# Common RT args, in kernel order.
CRT_IN0_ADDR, CRT_OUT_ADDR = range(2)


def _div_up(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class Placement:
    """BLOCK_SHARDED placement of the padded 2-D view, in tiles."""

    x0: int
    y0: int
    x1: int
    y1: int
    block_height_tiles: int
    block_width_tiles: int
    row_major: bool

    @classmethod
    def from_memory_config(cls, memory_config: ttnn.MemoryConfig) -> "Placement":
        spec = memory_config.shard_spec
        (rng,) = spec.grid.ranges()
        return cls(
            x0=rng.start.x,
            y0=rng.start.y,
            x1=rng.end.x,
            y1=rng.end.y,
            block_height_tiles=spec.shape[0] // TILE_DIM,
            block_width_tiles=spec.shape[1] // TILE_DIM,
            row_major=spec.orientation == ttnn.ShardOrientation.ROW_MAJOR,
        )

    def key(self) -> tuple:
        return (self.x0, self.y0, self.x1, self.y1, self.block_height_tiles, self.block_width_tiles, self.row_major)

    def block_core(self, r: int, c: int) -> tuple[int, int]:
        """Core (x, y) holding block (r, c)."""
        return (self.x0 + c, self.y0 + r) if self.row_major else (self.x0 + r, self.y0 + c)


@dataclass(frozen=True)
class Geometry:
    tensor_height_tiles: int
    tensor_width_tiles: int
    in0: Placement
    out: Placement
    aligned_page_bytes: int

    @property
    def num_block_rows(self) -> int:
        return _div_up(self.tensor_height_tiles, self.out.block_height_tiles)

    @property
    def num_block_cols(self) -> int:
        return _div_up(self.tensor_width_tiles, self.out.block_width_tiles)

    @property
    def out_shard_bytes(self) -> int:
        return self.out.block_height_tiles * self.out.block_width_tiles * self.aligned_page_bytes

    def holder_rectangle(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Inclusive core rectangle of out's block holders (the kernel range)."""
        nx, ny = (
            (self.num_block_cols, self.num_block_rows)
            if self.out.row_major
            else (self.num_block_rows, self.num_block_cols)
        )
        return (self.out.x0, self.out.y0), (self.out.x0 + nx - 1, self.out.y0 + ny - 1)

    def blocks(self):
        """Yield (r, c, core, row_begin, row_end, col_begin, col_end) of every out block, clipped to the tensor."""
        bh, bw = self.out.block_height_tiles, self.out.block_width_tiles
        for r in range(self.num_block_rows):
            row_begin = r * bh
            row_end = min(row_begin + bh, self.tensor_height_tiles)
            for c in range(self.num_block_cols):
                col_begin = c * bw
                col_end = min(col_begin + bw, self.tensor_width_tiles)
                yield r, c, self.out.block_core(r, c), row_begin, row_end, col_begin, col_end


def compute_geometry(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> Geometry:
    padded = list(input_tensor.padded_shape)
    height = 1
    for d in padded[:-1]:
        height *= d
    return Geometry(
        tensor_height_tiles=height // TILE_DIM,
        tensor_width_tiles=padded[-1] // TILE_DIM,
        in0=Placement.from_memory_config(input_tensor.memory_config()),
        out=Placement.from_memory_config(memory_config),
        aligned_page_bytes=input_tensor.buffer_aligned_page_size(),
    )


def issuer_row_ranges(row_begin: int, row_end: int, issuers_per_core: int = ISSUERS_PER_CORE):
    """Contiguous row range of each issuer: the first `rows % n` issuers take one row more."""
    rows = row_end - row_begin
    base, extra = divmod(rows, issuers_per_core)
    ranges = []
    begin = row_begin
    for k in range(issuers_per_core):
        end = begin + base + (1 if k < extra else 0)
        ranges.append((begin, end))
        begin = end
    return ranges


def piece_count(geom: Geometry, row_begin: int, row_end: int, col_begin: int, col_end: int) -> int:
    """Pieces = in0 blocks overlapping the issuer's sub-block (mirrors the kernel walk)."""
    if row_begin >= row_end or col_begin >= col_end:
        return 0
    ih, iw = geom.in0.block_height_tiles, geom.in0.block_width_tiles
    rows = (row_end - 1) // ih - row_begin // ih + 1
    cols = (col_end - 1) // iw - col_begin // iw + 1
    return rows * cols


def issuer_roles(geom: Geometry, issuers_per_core: int = ISSUERS_PER_CORE) -> list[dict]:
    """Role of every core in the kernel range: its out block and each issuer's rows and piece count (BB-21)."""
    roles = []
    for r, c, core, row_begin, row_end, col_begin, col_end in geom.blocks():
        issuers = []
        for k, (ib, ie) in enumerate(issuer_row_ranges(row_begin, row_end, issuers_per_core)):
            processor, noc = ISSUER_PLACEMENTS[k]
            issuers.append(
                {
                    "issuer": k,
                    "processor": str(processor),
                    "noc": str(noc),
                    "rows": (ib, ie),
                    "cols": (col_begin, col_end),
                    "pieces": piece_count(geom, ib, ie, col_begin, col_end),
                }
            )
        roles.append({"core": core, "out_block": (r, c), "issuers": issuers})
    return roles


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    geom: Geometry,
    issuers_per_core: int = ISSUERS_PER_CORE,
) -> ttnn.ProgramDescriptor:
    assert 1 <= issuers_per_core <= len(ISSUER_PLACEMENTS), issuers_per_core
    assert output_tensor.buffer_aligned_page_size() == geom.aligned_page_bytes, "in0 and out slot strides differ"

    (hx0, hy0), (hx1, hy1) = geom.holder_rectangle()
    holders = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(hx0, hy0), ttnn.CoreCoord(hx1, hy1))})

    ct_args = [
        geom.tensor_width_tiles,
        geom.out.block_width_tiles,
        geom.in0.block_height_tiles,
        geom.in0.block_width_tiles,
        geom.aligned_page_bytes,
    ]
    ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    common_rt_args = [0, 0]
    common_rt_args[CRT_IN0_ADDR] = input_tensor.buffer_address()
    common_rt_args[CRT_OUT_ADDR] = output_tensor.buffer_address()

    rt_args = [ttnn.RuntimeArgs() for _ in range(issuers_per_core)]
    for _r, _c, (x, y), row_begin, row_end, col_begin, col_end in geom.blocks():
        for k, (ib, ie) in enumerate(issuer_row_ranges(row_begin, row_end, issuers_per_core)):
            args = [0] * 5
            args[RT_BLOCK_ROW_BEGIN] = row_begin
            args[RT_ROW_BEGIN] = ib
            args[RT_ROW_END] = ie
            args[RT_COL_BEGIN] = col_begin
            args[RT_COL_END] = col_end
            rt_args[k][x][y] = args

    kernels = []
    for k in range(issuers_per_core):
        processor, noc = ISSUER_PLACEMENTS[k]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=ISSUER_KERNEL,
                core_ranges=holders,
                compile_time_args=ct_args,
                runtime_args=rt_args[k],
                common_runtime_args=common_rt_args,
                config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
            )
        )

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])


def set_buffer_addresses(program_descriptor: ttnn.ProgramDescriptor, in0_addr: int, out_addr: int) -> None:
    """Rewrite the per-call addresses of a cached descriptor (the only values that change on a cache hit)."""
    for kernel in program_descriptor.kernels:
        args = kernel.common_runtime_args  # bound by reference: writes land in the cached descriptor
        args[CRT_IN0_ADDR] = in0_addr
        args[CRT_OUT_ADDR] = out_addr
