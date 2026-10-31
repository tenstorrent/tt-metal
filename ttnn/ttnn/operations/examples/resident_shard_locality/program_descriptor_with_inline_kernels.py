# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Data already resident in L1: split the work by the shard spec, or ignore it.

Input and output are L1-sharded ROW_MAJOR bf16 or fp32 tensors (one page = one stick, or for
BLOCK_SHARDED one shard-width piece of a stick). The op is tilize -> untilize (a copy
through the tile format), so the compute is trivial and identical for every variant; what
changes is which Tensix core works on which sticks, and where the sticks come from.

    variant="resplit"     (baseline) the work split ignores the shard spec: the tensor is cut
                          into work units of 32 sticks (one tile-row) x one column chunk and the
                          units are dealt out evenly over the shard grid's Tensix cores, like an
                          interleaved tensor. Each Tensix core gathers its 32 sticks from
                          whichever Tensix cores own them (sharded TensorAccessor, one NoC read
                          per stick, one barrier per unit) and writes the untilized sticks back
                          to their owners the same way.
    variant="local"       the work split IS the shard spec: every Tensix core copies its own
                          shard's sticks with local NoC reads into a 2-chunk-deep staging CB,
                          tilizes, untilizes and copies the sticks back into its own output
                          shard. A shard height that is not a multiple of 32 sticks pads the
                          last tile-row locally, a shard width that is not a multiple of 32
                          elements pads the last tile-column locally (padding is never written
                          back). A wide stick is cut into column chunks of <= MAX_CHUNK_TILES
                          tiles and only the current chunk is staged, so per-Tensix-core L1 is
                          bounded by the chunk, not the row (see `local_l1_bytes`).
    variant="local_alias" `local` without the copies: the input CB is the input shard itself
                          and the output CB is the output shard (globally allocated CBs), so
                          tilize reads the resident sticks in place and untilize writes the
                          output sticks in place. Only legal when the shard is a whole number
                          of tile-rows (shard height % 32 == 0) and the shard width is a whole
                          number of tiles (shard width % 32 == 0), since then the shard's byte
                          layout equals a CB of whole tile-rows.

Reads/writes per stick, chunk widths, CB depths and the compute kernel are the same in all
variants; `resplit` vs `local` changes only the work split (and hence remote vs local NoC
traffic and how many tile-rows each Tensix core tilizes), `local` vs `local_alias` removes
only the two copies.
"""

import math

import ttnn

TILE = 32
MAX_CHUNK_TILES = 32  # widest column chunk (tiles) staged at once; bounds per-core L1 to 6 x 32 tiles
CB_IN, CB_TILED, CB_OUT = 0, 1, 16

VARIANTS = ("resplit", "local", "local_alias")
LAYOUTS = ("height", "block")
DTYPES = ("bfloat16", "float32")


def elem_bytes(dtype):
    """Bytes per element of a supported dtype (ttnn.bfloat16 -> 2, ttnn.float32 -> 4)."""
    return {ttnn.bfloat16: 2, ttnn.float32: 4}[dtype]


def tile_bytes(dtype):
    """Tile size in bytes of one 32x32 tile (bf16 2048 B, fp32 4096 B)."""
    return TILE * TILE * elem_bytes(dtype)


# ---------------------------------------------------------------------------------------
# Kernels (inline)
# ---------------------------------------------------------------------------------------

_COMPUTE = r"""
#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

// tilize -> untilize, one block (= one tile-row x ct tiles) at a time. Identical for every variant.
// fp32: Lossless tilize + UnpackToDestFp32 on both CBs + fp32 Dest keep the copy bit-exact
// (Lossless has no effect on bf16).
void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_tiled = 1;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t ct = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    using namespace compute_kernel_lib;
    compute_kernel_hw_startup(cb_in, cb_tiled);
    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t b = 0; b < num_blocks; ++b) {
            tilize<ct, cb_in, cb_tiled,
                   tilize_config::InitUninitMode::InitAndUninit,
                   tilize_config::WaitMode::WaitBlock,
                   tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
                   tilize_config::Fp32Mode::Lossless>(1);
            untilize<ct, cb_tiled, cb_out,
                     untilize_config::InitUninitMode::InitAndUninit,
                     untilize_config::WaitMode::WaitBlock,
                     untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
        }
    }
}
"""

# Shared compile-time args of the four copy kernels:
#   0 page_bytes  (aligned bytes of one page = one stick piece)
#   1 ct          (tiles per column chunk)
#   2 num_chunks  (column chunks per page)
#   3 row_bytes   (valid bytes of one page)
#   4 kernel_iters
#   5 slice_bytes (bytes of one 32-element stick slice: 64 bf16, 128 fp32)
_COPY_PRELUDE = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
constexpr uint32_t ct = get_compile_time_arg_val(1);
constexpr uint32_t num_chunks = get_compile_time_arg_val(2);
constexpr uint32_t row_bytes = get_compile_time_arg_val(3);
constexpr uint32_t kernel_iters = get_compile_time_arg_val(4);
constexpr uint32_t slice_bytes = get_compile_time_arg_val(5);
constexpr uint32_t chunk_stride = ct * slice_bytes;  // bytes of one staged stick slice (ct tiles wide)
// The last chunk may be partial (fewer valid tiles, a partial tile-column): only its valid bytes move;
// the rest of the staged slot is padding that is tilized and never written back.
constexpr uint32_t last_chunk_bytes = row_bytes - (num_chunks - 1) * chunk_stride;

FORCE_INLINE uint32_t chunk_bytes(uint32_t k) { return k + 1 == num_chunks ? last_chunk_bytes : chunk_stride; }
"""

# local: own shard, base + offset addressing, local NoC reads.
_LOCAL_READER = (
    _COPY_PRELUDE
    + r"""
void kernel_main() {
    constexpr uint32_t cb_in = 0;
    const uint32_t shard_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);  // sticks in this Tensix core's shard

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t r0 = 0; r0 < num_rows; r0 += 32) {
            const uint32_t rows = num_rows - r0 < 32 ? num_rows - r0 : 32;  // < 32: last tile-row padded locally
            for (uint32_t k = 0; k < num_chunks; ++k) {
                const uint32_t bytes = chunk_bytes(k);
                cb_reserve_back(cb_in, ct);
                uint32_t l1 = get_write_ptr(cb_in);
                uint32_t src = shard_addr + r0 * page_bytes + k * chunk_stride;
                for (uint32_t i = 0; i < rows; ++i) {
                    noc_async_read(get_noc_addr(src), l1, bytes);
                    src += page_bytes;
                    l1 += chunk_stride;
                }
                noc_async_read_barrier();
                cb_push_back(cb_in, ct);
            }
        }
    }
}
"""
)

_LOCAL_WRITER = (
    _COPY_PRELUDE
    + r"""
void kernel_main() {
    constexpr uint32_t cb_out = 16;
    const uint32_t shard_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t r0 = 0; r0 < num_rows; r0 += 32) {
            const uint32_t rows = num_rows - r0 < 32 ? num_rows - r0 : 32;  // padding rows are dropped
            for (uint32_t k = 0; k < num_chunks; ++k) {
                const uint32_t bytes = chunk_bytes(k);
                cb_wait_front(cb_out, ct);
                uint32_t l1 = get_read_ptr(cb_out);
                uint32_t dst = shard_addr + r0 * page_bytes + k * chunk_stride;
                for (uint32_t i = 0; i < rows; ++i) {
                    noc_async_write(l1, get_noc_addr(dst), bytes);
                    dst += page_bytes;
                    l1 += chunk_stride;
                }
                noc_async_write_barrier();
                cb_pop_front(cb_out, ct);
            }
        }
    }
}
"""
)

# resplit: work units (tile-row, page column, chunk) dealt out like an interleaved tensor;
# each stick piece is fetched from its owner through the sharded TensorAccessor.
#   6 total_rows (sticks in the tensor)   7 page_cols (pages per stick: 1 height, shard columns block)
_RESPLIT_UNIT = r"""
constexpr uint32_t total_rows = get_compile_time_arg_val(6);
constexpr uint32_t page_cols = get_compile_time_arg_val(7);
constexpr auto acc_args = TensorAccessorArgs<8>();
"""

_RESPLIT_READER = (
    _COPY_PRELUDE
    + _RESPLIT_UNIT
    + r"""
void kernel_main() {
    constexpr uint32_t cb_in = 0;
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t unit_start = get_arg_val<uint32_t>(1);
    const uint32_t unit_count = get_arg_val<uint32_t>(2);
    const auto acc = TensorAccessor(acc_args, addr, page_bytes);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
            const uint32_t k = u % num_chunks;
            const uint32_t pc = (u / num_chunks) % page_cols;
            const uint32_t r0 = (u / num_chunks / page_cols) * 32;
            const uint32_t rows = total_rows - r0 < 32 ? total_rows - r0 : 32;
            const uint32_t bytes = chunk_bytes(k);
            cb_reserve_back(cb_in, ct);
            uint32_t l1 = get_write_ptr(cb_in);
            for (uint32_t i = 0; i < rows; ++i) {
                noc_async_read(acc.get_noc_addr((r0 + i) * page_cols + pc, k * chunk_stride), l1, bytes);
                l1 += chunk_stride;
            }
            noc_async_read_barrier();
            cb_push_back(cb_in, ct);
        }
    }
}
"""
)

_RESPLIT_WRITER = (
    _COPY_PRELUDE
    + _RESPLIT_UNIT
    + r"""
void kernel_main() {
    constexpr uint32_t cb_out = 16;
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t unit_start = get_arg_val<uint32_t>(1);
    const uint32_t unit_count = get_arg_val<uint32_t>(2);
    const auto acc = TensorAccessor(acc_args, addr, page_bytes);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
            const uint32_t k = u % num_chunks;
            const uint32_t pc = (u / num_chunks) % page_cols;
            const uint32_t r0 = (u / num_chunks / page_cols) * 32;
            const uint32_t rows = total_rows - r0 < 32 ? total_rows - r0 : 32;
            const uint32_t bytes = chunk_bytes(k);
            cb_wait_front(cb_out, ct);
            uint32_t l1 = get_read_ptr(cb_out);
            for (uint32_t i = 0; i < rows; ++i) {
                noc_async_write(l1, acc.get_noc_addr((r0 + i) * page_cols + pc, k * chunk_stride), bytes);
                l1 += chunk_stride;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, ct);
        }
    }
}
"""
)

# local_alias: the CBs ARE the shards; the data-movement kernels only publish / retire them.
_ALIAS_READER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);  // whole input shard, in tile-sized pages
    for (uint32_t it = 0; it < kernel_iters; ++it) {
        cb_reserve_back(cb_in, num_tiles);
        cb_push_back(cb_in, num_tiles);
    }
}
"""

_ALIAS_WRITER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    for (uint32_t it = 0; it < kernel_iters; ++it) {
        cb_wait_front(cb_out, num_tiles);
        cb_pop_front(cb_out, num_tiles);
    }
}
"""


# ---------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------


def shard_grid_cores(device, layout, cores):
    """Tensix cores of the shard grid in shard order (ROW_MAJOR orientation).

    layout="height": `cores` is a count; filled row-major across the compute grid.
    layout="block":  `cores` is (rows, cols); a rectangle at the grid origin."""
    grid = device.compute_with_storage_grid_size()
    if layout == "height":
        if not 1 <= cores <= grid.x * grid.y:
            raise ValueError(f"resident_shard_locality: {cores} Tensix cores do not fit the {grid.x}x{grid.y} grid")
        return [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(cores)]
    rows, cols = cores
    if rows > grid.y or cols > grid.x:
        raise ValueError(f"resident_shard_locality: block grid {rows}x{cols} does not fit {grid.x}x{grid.y}")
    return [ttnn.CoreCoord(c, r) for r in range(rows) for c in range(cols)]


def _core_range_set(device, layout, cores):
    grid = device.compute_with_storage_grid_size()
    if layout == "height":
        return ttnn.num_cores_to_corerangeset(cores, grid, True)
    rows, cols = cores
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))])


def sharded_memory_config(device, layout, cores, shard_rows, shard_width):
    """L1 HEIGHT_SHARDED (cores = count) or BLOCK_SHARDED (cores = (rows, cols)) memory config."""
    memory_layout = {
        "height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "block": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    }[layout]
    spec = ttnn.ShardSpec(
        _core_range_set(device, layout, cores), [shard_rows, shard_width], ttnn.ShardOrientation.ROW_MAJOR
    )
    return ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, spec)


def tensor_shape(layout, cores, shard_rows, shard_width):
    """Logical [sticks, elements] of the whole tensor."""
    if layout == "height":
        return [cores * shard_rows, shard_width]
    rows, cols = cores
    return [rows * shard_rows, cols * shard_width]


def chunk_plan(shard_width):
    """(tiles per column chunk, column chunks per page) for a page of `shard_width` elements.

    Fewest chunks of <= MAX_CHUNK_TILES tiles, balanced: chunks = ceil(width tiles / 32), chunk =
    ceil(width tiles / chunks). The last chunk may carry up to chunks - 1 padding tiles (e.g. 127
    tiles -> 4 chunks of 32, the last with 31 valid tiles), never a 1-tile chunk for a prime width."""
    width_tiles = math.ceil(shard_width / TILE)
    num_chunks = math.ceil(width_tiles / MAX_CHUNK_TILES)
    return math.ceil(width_tiles / num_chunks), num_chunks


def chunk_tiles(shard_width):
    return chunk_plan(shard_width)[0]


def local_l1_bytes(shard_width, dtype, *, chunked=True):
    """Per-Tensix-core CB bytes of `local` (3 CBs x 2 chunks deep, tile-sized pages), excluding the shards.

    chunked=False: what staging the whole row (one chunk = the whole padded width) would need."""
    ct = chunk_tiles(shard_width) if chunked else math.ceil(shard_width / TILE)
    return 3 * 2 * ct * tile_bytes(dtype)


def alias_eligible(shard_rows, shard_width):
    return shard_rows % TILE == 0 and shard_width % TILE == 0 and shard_width // TILE <= MAX_CHUNK_TILES


def _geometry(input_tensor):
    mc = input_tensor.memory_config()
    if not mc.is_sharded() or mc.buffer_type != ttnn.BufferType.L1:
        raise ValueError("resident_shard_locality: input must be L1-sharded")
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT or input_tensor.dtype not in (ttnn.bfloat16, ttnn.float32):
        raise ValueError("resident_shard_locality: input must be ROW_MAJOR bfloat16 or float32")
    if mc.memory_layout not in (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED):
        raise ValueError("resident_shard_locality: only HEIGHT_SHARDED and BLOCK_SHARDED are supported")
    spec = mc.shard_spec
    if spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
        raise ValueError("resident_shard_locality: only ROW_MAJOR shard orientation is supported")
    shard_rows, shard_width = spec.shape
    total_rows, width = list(input_tensor.shape)
    page_cols = width // shard_width
    if total_rows % shard_rows or width % shard_width:
        raise ValueError("resident_shard_locality: tensor must be a whole number of full shards")
    return spec, shard_rows, shard_width, total_rows, page_cols


def create_program_descriptor(input_tensor, output_tensor, *, variant, kernel_iters=1, swap_nocs=False):
    """swap_nocs: ablation knob - reader on NoC1 / writer on NoC0 instead of the default NoC0 / NoC1."""
    if variant not in VARIANTS:
        raise ValueError(f"resident_shard_locality: variant must be one of {VARIANTS}, got {variant!r}")
    spec, shard_rows, shard_width, total_rows, page_cols = _geometry(input_tensor)
    grid = spec.grid
    cores = ttnn.corerange_to_cores(grid, None, True)  # shard order (ROW_MAJOR)
    page_bytes = input_tensor.buffer_aligned_page_size()
    dtype = input_tensor.dtype
    esize, tbytes = elem_bytes(dtype), tile_bytes(dtype)
    row_bytes = shard_width * esize

    if variant == "local_alias":
        if not alias_eligible(shard_rows, shard_width):
            raise ValueError(
                "resident_shard_locality: local_alias needs shard height % 32 == 0 and shard width % 32 == 0 "
                f"(<= {MAX_CHUNK_TILES} tiles), got [{shard_rows}, {shard_width}]"
            )
        ct = shard_width // TILE
        num_tiles = shard_rows // TILE * ct
        cb_in = ttnn.cb_descriptor_from_sharded_tensor(CB_IN, input_tensor)
        cb_out = ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor)
        for cb in (cb_in, cb_out):  # the shard's bytes, re-viewed as tile-sized CB pages
            assert cb.total_size == num_tiles * tbytes, (cb.total_size, num_tiles)
            cb.format_descriptors[0].page_size = tbytes
        cb_tiled = _cb(CB_TILED, 2 * ct, grid, dtype)
        rt = {"reader": ttnn.RuntimeArgs(), "writer": ttnn.RuntimeArgs(), "compute": ttnn.RuntimeArgs()}
        for core in cores:
            rt["reader"][core.x][core.y] = [num_tiles]
            rt["writer"][core.x][core.y] = [num_tiles]
            rt["compute"][core.x][core.y] = [shard_rows // TILE]
        reader_src, writer_src = _ALIAS_READER, _ALIAS_WRITER
        reader_ct = writer_ct = [kernel_iters]
        cbs = [cb_in, cb_tiled, cb_out]
    else:
        ct, num_chunks = chunk_plan(shard_width)
        cbs = [_cb(CB_IN, 2 * ct, grid, dtype), _cb(CB_TILED, 2 * ct, grid, dtype), _cb(CB_OUT, 2 * ct, grid, dtype)]
        dm_ct = [page_bytes, ct, num_chunks, row_bytes, kernel_iters, TILE * esize]
        rt = {"reader": ttnn.RuntimeArgs(), "writer": ttnn.RuntimeArgs(), "compute": ttnn.RuntimeArgs()}
        if variant == "local":
            blocks = math.ceil(shard_rows / TILE) * num_chunks
            for core in cores:
                rt["reader"][core.x][core.y] = [input_tensor.buffer_address(), shard_rows]
                rt["writer"][core.x][core.y] = [output_tensor.buffer_address(), shard_rows]
                rt["compute"][core.x][core.y] = [blocks]
            reader_src, writer_src = _LOCAL_READER, _LOCAL_WRITER
            reader_ct = writer_ct = dm_ct
        else:  # resplit
            units = math.ceil(total_rows / TILE) * page_cols * num_chunks
            base, rem = divmod(units, len(cores))
            start = 0
            for j, core in enumerate(cores):
                count = base + (1 if j < rem else 0)
                rt["reader"][core.x][core.y] = [input_tensor.buffer_address(), start, count]
                rt["writer"][core.x][core.y] = [output_tensor.buffer_address(), start, count]
                rt["compute"][core.x][core.y] = [count]
                start += count
            reader_src, writer_src = _RESPLIT_READER, _RESPLIT_WRITER
            unit_ct = dm_ct + [total_rows, page_cols]
            reader_ct = unit_ct + list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
            writer_ct = unit_ct + list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_cfg, writer_cfg = ttnn.ReaderConfigDescriptor(), ttnn.WriterConfigDescriptor()
    if swap_nocs:
        reader_cfg = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)
        writer_cfg = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)

    def kernel(src, ct_args, runtime_args, config):
        return ttnn.KernelDescriptor(
            kernel_source=src,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=grid,
            compile_time_args=ct_args,
            runtime_args=runtime_args,
            config=config,
        )

    return ttnn.ProgramDescriptor(
        kernels=[
            kernel(reader_src, reader_ct, rt["reader"], reader_cfg),
            kernel(writer_src, writer_ct, rt["writer"], writer_cfg),
            kernel(_COMPUTE, [ct, kernel_iters], rt["compute"], _compute_config(dtype)),
        ],
        semaphores=[],
        cbs=cbs,
    )


def _cb(index, num_tiles, grid, dtype):
    return ttnn.CBDescriptor(
        total_size=num_tiles * tile_bytes(dtype),
        core_ranges=grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=tile_bytes(dtype))
        ],
    )


def _compute_config(dtype):
    """bf16: defaults. fp32: fp32 Dest + UnpackToDestFp32 on the two unpacked CBs (bit-exact copy)."""
    cfg = ttnn.ComputeConfigDescriptor()
    if dtype == ttnn.float32:
        cfg.fp32_dest_acc_en = True
        modes = [ttnn.UnpackToDestMode.Default] * 64  # one entry per circular buffer
        modes[CB_IN] = modes[CB_TILED] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes
    return cfg


def resident_shard_copy(input_tensor, *, variant="local", kernel_iters=1, swap_nocs=False):
    """tilize -> untilize of an L1-sharded ROW_MAJOR bf16/fp32 tensor; output has the input's memory config.

    variant: "resplit" (work split ignores the shard spec), "local" (work split = shard spec,
    local copies) or "local_alias" (local, CBs aliased onto the shards; whole-tile shards only).
    swap_nocs: ablation only - run the reader on NoC1 and the writer on NoC0."""
    if kernel_iters < 1:
        raise ValueError(f"resident_shard_locality: kernel_iters must be >= 1, got {kernel_iters}")
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        ttnn.ROW_MAJOR_LAYOUT,
        input_tensor.device(),
        input_tensor.memory_config(),
    )
    descriptor = create_program_descriptor(
        input_tensor, output_tensor, variant=variant, kernel_iters=kernel_iters, swap_nocs=swap_nocs
    )
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
