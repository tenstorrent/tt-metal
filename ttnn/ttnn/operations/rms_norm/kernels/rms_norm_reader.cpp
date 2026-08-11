// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm reader (NCRISC / NoC0).
//
// Per kernel, once:
//   prepare_constants()  — cb_scaler (reduce scaler 1.0), cb_wmask (0/1 column
//                          mask for the ragged hidden tile), cb_zero_tile.
//   load_gamma_slice()   — cb_gamma_tiles: this core's gamma slice, resident for
//                          the whole kernel (never popped by compute), so gamma
//                          crosses DRAM once per core.
//
// Per block (a block_row_tiles x cb_w_tiles tile rectangle):
//   load_block()         — the whole block behind ONE read barrier on the tiled
//                          path, or one barrier per 32-row block on the
//                          ROW_MAJOR path (read_slice_rows).
//
// PLACEMENT. On an INTERLEAVED input, load_block reads DRAM through a
// TensorAccessor. On a physically SHARDED input the block is ALREADY in this
// core's L1 and there is NO NoC read of it:
//   * TILE shard      — cb_input_tiles is pinned zero-copy over the shard buffer
//                       (ttnn.cb_descriptor_from_sharded_tensor), so load_block is
//                       the CB handshake alone.
//   * ROW_MAJOR shard — the block CB is the tilize staging buffer, whose uniform
//                       tile-row stride is not the shard's stick stride, so the
//                       sticks are re-strided CORE-LOCALLY (read_shard_rows,
//                       L1 -> L1 via LocalShardAccessor). Still no DRAM crossing.
// A core's own shard is NEVER addressed through a TensorAccessor.
//
// UNIFORM BLOCK WIDTH. Every CB moves CB_W_TILES pages per tile-row on EVERY
// core, while a core's VALID hidden slice is `core_w <= CB_W_TILES` (the ragged
// remainder of the hidden split). Two hard constraints force this:
//   * cb_rstd / cb_stat_gather are addressed by a peer's LOCAL pointer, so the
//     L1 map must be identical group-wide;
//   * a CB's capacity must be an exact multiple of its push/pop quantum
//     (dataflow_api.h:216-221, "no other wrap is legal").
// The (CB_W_TILES - core_w) trailing pad pages are pushed but never read by the
// statistics phases, which walk `core_w` columns at row stride CB_W_TILES.
//
// UNINITIALIZED L1 IN THE BLOCK, and the invariant that makes it safe. Two
// regions of a pushed block are never written by this reader: the pad columns
// above, and the stale rows of a ragged final 32-row group (H % 32 != 0). Both
// DO reach the FPU (tilize and the apply phases cover the whole uniform block).
// This is safe ONLY because every math phase over this block is row-independent:
// the statistics are a REDUCE_ROW (each output row folds only its own row's
// columns) and the apply phases are element-wise, so a stale Inf/NaN cannot
// migrate into a valid row, and the writer never stores those bytes to DRAM.
// The pad COLUMNS inside the last VALID hidden tile are a different matter and
// are NOT left to chance — mask_tail_block zeroes them numerically, which is
// what the pad_poison cases test. If a phase over cb_input_tiles ever crosses
// rows or columns (a REDUCE_COL / REDUCE_SCALAR / DestAccumulation::WholeShape),
// this reader must zero-fill both regions first.
//
// Helper substitutions (raw NoC instead of a kernel_lib helper), with reasons:
//   * load_block on the TILE path uses raw noc_async_read_tile over a
//     TensorAccessor. read_sticks_for_tilize is ROW-MAJOR ONLY: it derives a
//     stick stride and asserts tile_size % tile_hw == 0
//     (tilize_helpers_dataflow.inl:82-85), and a TILE tensor's DRAM pages are
//     already tiles, so it has no sticks to read.
//   * load_block on the RM path also uses raw NoC. read_sticks_for_tilize
//     derives BOTH its page count and its L1 row stride from `row_bytes`
//     (tilize_helpers_dataflow.inl:91-93,120-124), so it can only produce a
//     block whose row stride equals the core's own valid slice. tilize<CB_W>
//     consumes a 32 x (CB_W*32) row-major block, i.e. the GROUP-UNIFORM stride,
//     which is wider than the valid slice on a ragged core — the helper cannot
//     express that shape. read_slice_rows() below is the helper's body with the
//     stride taken from the uniform CB width instead of from row_bytes.
//   * cb_zero_tile is filled with a direct L1 memset: it is the identity operand
//     of the combine's DEST accumulation, not a reduce scaler, so the
//     reduce-scaler helpers (whose contract is "reduce LLK only") do not apply.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_input_tiles = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_wmask = 3;
constexpr uint32_t cb_zero_tile = 4;
constexpr uint32_t cb_gamma_rm = 12;
constexpr uint32_t cb_gamma_tiles = 13;
constexpr uint32_t TILE_HW_DIM = 32;

// Zero `bytes` of L1 at `addr`. Used for the pad COLUMNS of a ragged hidden slice
// (see read_slice_rows): those columns are multiplied by a 0/1 mask before the
// square, and `garbage * 0` is NaN if the garbage happens to be non-finite, which
// would poison that row's Sum x^2. A ROW_MAJOR WIDTH/BLOCK shard makes this the
// common case, not the corner one: its width granule is the L1 alignment, so a
// core's slice can be 16 elements of a 32-column tile.
FORCE_INLINE void zero_l1_bytes(uint32_t addr, uint32_t bytes) {
    if (bytes == 0) {
        return;
    }
    if (((addr | bytes) & 3u) == 0) {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
        for (uint32_t i = 0, n = bytes >> 2; i < n; ++i) {
            p[i] = 0;
        }
    } else {
        volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr);
        for (uint32_t i = 0, n = bytes >> 1; i < n; ++i) {
            p[i] = 0;
        }
    }
}

// A resident-shard "accessor" with the same page/offset shape as TensorAccessor,
// resolving to THIS core's own L1: the shard is already here, so the address is
// `shard_base + page * shard_row_bytes`. This is what lets read_slice_rows (and
// the writer's write_slice_rows) serve the DRAM leg and the resident-shard leg
// with ONE body — the sharded leg differs only in where the address comes from.
struct LocalShardAccessor {
    uint32_t base;
    uint32_t row_bytes;
    FORCE_INLINE uint64_t get_noc_addr(uint32_t page, uint32_t byte_offset) const {
        return ::get_noc_addr(base + page * row_bytes + byte_offset);
    }
};

// Read `rows` row-major sticks of `slice_bytes` into one uniform-width CB block
// (`cb_w_tiles` tile-sized pages), laying each stick at the block's uniform L1
// row stride. The trailing (cb_w_tiles*32*elem - slice_bytes) pad bytes of each
// row are ZEROED (see zero_l1_bytes) — free when the slice fills the block row,
// which is every tile-aligned case. Stale rows of a ragged 32-row group are left
// untouched: every math phase over this block is row-independent, so their
// garbage cannot migrate into a valid row, and the writer never stores them.
template <uint32_t cb_id, uint32_t cb_w_tiles, typename Accessor>
FORCE_INLINE void read_slice_rows(
    const Accessor& acc, uint32_t rows, uint32_t slice_bytes, uint32_t start_page, uint32_t byte_offset) {
    constexpr uint32_t tile_row_bytes = get_tile_size(cb_id) / TILE_HW_DIM;
    constexpr uint32_t block_row_bytes = tile_row_bytes * cb_w_tiles;
    const uint32_t pad_bytes = block_row_bytes - slice_bytes;
    cb_reserve_back(cb_id, cb_w_tiles);
    uint32_t l1_addr = get_write_ptr(cb_id);
    for (uint32_t r = 0; r < rows; ++r) {
        noc_async_read(acc.get_noc_addr(start_page + r, byte_offset), l1_addr, slice_bytes);
        zero_l1_bytes(l1_addr + slice_bytes, pad_bytes);
        l1_addr += block_row_bytes;
    }
    // One barrier per 32-row block == cb_w_tiles tiles per barrier.
    noc_async_read_barrier();
    cb_push_back(cb_id, cb_w_tiles);
}

// The resident-shard flavour of read_slice_rows: a core-local L1 re-stride of its
// OWN shard's sticks into the group-uniform tile-row stride tilize<> requires.
// There is no DRAM crossing on this path. When the shard's stick stride already
// IS the block row stride (a tile-aligned hidden slice, i.e. every HEIGHT-sharded
// ROW_MAJOR case with W % 32 == 0) the whole 32-row group is ONE transfer instead
// of 32; otherwise it falls back to the per-stick body above.
template <uint32_t cb_id, uint32_t cb_w_tiles>
FORCE_INLINE void read_shard_rows(
    uint32_t base, uint32_t shard_row_bytes, uint32_t rows, uint32_t slice_bytes, uint32_t start_page) {
    constexpr uint32_t block_row_bytes = (get_tile_size(cb_id) / TILE_HW_DIM) * cb_w_tiles;
    if (slice_bytes == block_row_bytes && shard_row_bytes == block_row_bytes) {
        cb_reserve_back(cb_id, cb_w_tiles);
        noc_async_read(
            ::get_noc_addr(base + start_page * shard_row_bytes), get_write_ptr(cb_id), rows * block_row_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_id, cb_w_tiles);
        return;
    }
    read_slice_rows<cb_id, cb_w_tiles>(LocalShardAccessor{base, shard_row_bytes}, rows, slice_bytes, start_page, 0);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t CB_W_TILES = get_compile_time_arg_val(0);
    constexpr uint32_t TENSOR_W_TILES = get_compile_time_arg_val(1);
    constexpr bool IS_RM_IN = get_compile_time_arg_val(2) != 0;
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(3) != 0;
    constexpr bool IS_RM_GAMMA = get_compile_time_arg_val(4) != 0;
    constexpr bool HAS_ANY_TAIL = get_compile_time_arg_val(5) != 0;
    constexpr bool IS_SHARDED_IN = get_compile_time_arg_val(6) != 0;
    constexpr auto src_args = TensorAccessorArgs<7>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t row_tile_start = get_arg_val<uint32_t>(2);
    const uint32_t num_blocks = get_arg_val<uint32_t>(3);
    const uint32_t block_row_tiles = get_arg_val<uint32_t>(4);
    const uint32_t last_block_row_tiles = get_arg_val<uint32_t>(5);
    const uint32_t w_tile_start = get_arg_val<uint32_t>(6);
    const uint32_t core_w = get_arg_val<uint32_t>(7);
    const uint32_t core_partial_w = get_arg_val<uint32_t>(8);
    const uint32_t num_sticks = get_arg_val<uint32_t>(9);
    const uint32_t stick_start = get_arg_val<uint32_t>(10);
    const uint32_t in_slice_bytes = get_arg_val<uint32_t>(11);
    const uint32_t in_byte_offset = get_arg_val<uint32_t>(12);
    [[maybe_unused]] const uint32_t gamma_slice_bytes = get_arg_val<uint32_t>(13);
    [[maybe_unused]] const uint32_t gamma_read_offset = get_arg_val<uint32_t>(14);
    [[maybe_unused]] const uint32_t gamma_lead_bytes = get_arg_val<uint32_t>(15);
    [[maybe_unused]] const uint32_t shard_row_bytes = get_arg_val<uint32_t>(16);

    // A mcast-box FILLER core: inside a reduction group's broadcast rectangle (a
    // shard grid is not always a rectangle) but owning no shard, so it carries no
    // work at all. It stays a program core only so the group's L1 map is uniform.
    if (num_blocks == 0) {
        return;
    }

    // ---------------- prepare_constants (once per kernel) ----------------
    // Pool-type-aware overload: SUM/REDUCE_ROW fills the matmul-path scaler
    // layout. The masking of the ragged hidden tile is done numerically by the
    // compute kernel, so the scaler here is a plain 1.0.
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    if constexpr (HAS_ANY_TAIL) {
        if (core_partial_w != 0) {
            // 1.0 in columns [0, core_partial_w), 0 elsewhere, in the row-0
            // broadcast layout the compute kernel consumes with BroadcastDim::Row.
            // PER-CORE, not per-tensor: a ROW_MAJOR WIDTH/BLOCK shard's width
            // granule is the L1 alignment, so EVERY core's hidden slice can end
            // mid-tile, not just the one owning the tensor's last tile.
            dataflow_kernel_lib::prepare_reduce_mask<cb_wmask, ckernel::ReduceDim::REDUCE_ROW>(core_partial_w);
        }
    }

    {
        cb_reserve_back(cb_zero_tile, 1);
        const uint32_t zero_addr = get_write_ptr(cb_zero_tile);
        const uint32_t words = get_tile_size(cb_zero_tile) / 4;
        volatile tt_l1_ptr uint32_t* zp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_addr);
        for (uint32_t i = 0; i < words; ++i) {
            zp[i] = 0;
        }
        cb_push_back(cb_zero_tile, 1);
    }

    // ---------------- load_gamma_slice (once per kernel) -----------------
    if constexpr (HAS_GAMMA) {
        if constexpr (IS_RM_GAMMA) {
            // One stick; compute tilizes it into the row-0-valid tile form that
            // TILE gamma already has.
            const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr);
            if (gamma_lead_bytes == 0) {
                read_slice_rows<cb_gamma_rm, CB_W_TILES>(gamma_acc, 1, gamma_slice_bytes, 0, gamma_read_offset);
            } else {
                // This core's gamma slice does NOT start on a DRAM-aligned boundary
                // (a ROW_MAJOR WIDTH/BLOCK shard's width granule is the L1
                // alignment, so a slice can start mid-tile). A DRAM read TRUNCATES
                // its source address to the DRAM alignment — silently returning a
                // neighbouring slice — so fetch the aligned span into scratch past
                // the gamma row, then re-fetch it L1 -> L1 at the exact offset: an
                // L1 source only needs the 16-byte L1 alignment, which every slice
                // offset has (the RM width granule IS that alignment).
                constexpr uint32_t block_row_bytes = (get_tile_size(cb_gamma_rm) / TILE_HW_DIM) * CB_W_TILES;
                cb_reserve_back(cb_gamma_rm, CB_W_TILES);
                const uint32_t row = get_write_ptr(cb_gamma_rm);
                // Scratch lives in tile-rows 1.. of the same CB block, which tilize
                // does copy into the gamma tile's rows 1..31 — harmless, because
                // gamma_block consumes row 0 only (BroadcastDim::Row).
                const uint32_t scratch = row + block_row_bytes;
                noc_async_read(
                    gamma_acc.get_noc_addr(0, gamma_read_offset), scratch, gamma_lead_bytes + gamma_slice_bytes);
                noc_async_read_barrier();
                noc_async_read(::get_noc_addr(scratch + gamma_lead_bytes), row, gamma_slice_bytes);
                zero_l1_bytes(row + gamma_slice_bytes, block_row_bytes - gamma_slice_bytes);
                noc_async_read_barrier();
                cb_push_back(cb_gamma_rm, CB_W_TILES);
            }
        } else {
            const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma_tiles);
            const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr, gamma_tile_bytes);
            cb_reserve_back(cb_gamma_tiles, CB_W_TILES);
            const uint32_t dst = get_write_ptr(cb_gamma_tiles);
            for (uint32_t c = 0; c < core_w; ++c) {
                noc_async_read_tile(w_tile_start + c, gamma_acc, dst + c * gamma_tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_gamma_tiles, CB_W_TILES);
        }
    }

    // ---------------- load_block (per block) -----------------------------
    if constexpr (IS_RM_IN && IS_SHARDED_IN) {
        // Resident ROW_MAJOR shard: no DRAM leg. The shard's stick stride is its own
        // width, not the group-uniform tile-row stride tilize<CB_W_TILES> consumes,
        // so the sticks are re-strided CORE-LOCALLY (L1 -> L1) instead of being
        // pinned. `stick_start` is 0 — a core addresses its own shard's page 0.
        uint32_t sticks_done = 0;
        for (uint32_t b = 0; b < num_blocks; ++b) {
            const uint32_t rows_t = (b + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles;
            for (uint32_t r = 0; r < rows_t; ++r) {
                uint32_t sticks_this = TILE_HW_DIM;
                if (sticks_this > num_sticks - sticks_done) {
                    sticks_this = num_sticks - sticks_done;
                }
                read_shard_rows<cb_input_rm, CB_W_TILES>(
                    src_addr, shard_row_bytes, sticks_this, in_slice_bytes, stick_start + sticks_done);
                sticks_done += sticks_this;
            }
        }
    } else if constexpr (IS_SHARDED_IN) {
        // Resident TILE shard: cb_input_tiles is PINNED zero-copy over it, so the
        // block is already in place and "load_block" is the CB handshake alone —
        // zero NoC traffic. The shard's pages are tile-row-major at row stride
        // CB_W_TILES, exactly the block layout compute expects, and the shard holds
        // >= the whole row assignment, so the per-block push walks straight through
        // it and never wraps.
        for (uint32_t b = 0; b < num_blocks; ++b) {
            const uint32_t rows_t = (b + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles;
            const uint32_t pages = rows_t * CB_W_TILES;
            cb_reserve_back(cb_input_tiles, pages);
            cb_push_back(cb_input_tiles, pages);
        }
    } else if constexpr (IS_RM_IN) {
        const auto acc = TensorAccessor(src_args, src_addr);
        uint32_t sticks_done = 0;
        for (uint32_t b = 0; b < num_blocks; ++b) {
            const uint32_t rows_t = (b + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles;
            for (uint32_t r = 0; r < rows_t; ++r) {
                uint32_t sticks_this = TILE_HW_DIM;
                if (sticks_this > num_sticks - sticks_done) {
                    sticks_this = num_sticks - sticks_done;
                }
                read_slice_rows<cb_input_rm, CB_W_TILES>(
                    acc, sticks_this, in_slice_bytes, stick_start + sticks_done, in_byte_offset);
                sticks_done += sticks_this;
            }
        }
    } else {
        const uint32_t tile_bytes = get_tile_size(cb_input_tiles);
        const auto acc = TensorAccessor(src_args, src_addr, tile_bytes);
        for (uint32_t b = 0; b < num_blocks; ++b) {
            const uint32_t rows_t = (b + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles;
            const uint32_t pages = rows_t * CB_W_TILES;
            cb_reserve_back(cb_input_tiles, pages);
            const uint32_t dst = get_write_ptr(cb_input_tiles);
            for (uint32_t r = 0; r < rows_t; ++r) {
                const uint32_t row_tile = row_tile_start + b * block_row_tiles + r;
                const uint32_t base = row_tile * TENSOR_W_TILES + w_tile_start;
                for (uint32_t c = 0; c < core_w; ++c) {
                    noc_async_read_tile(base + c, acc, dst + (r * CB_W_TILES + c) * tile_bytes);
                }
            }
            // The whole block behind one barrier — never one barrier per tile.
            noc_async_read_barrier();
            cb_push_back(cb_input_tiles, pages);
        }
    }
}
