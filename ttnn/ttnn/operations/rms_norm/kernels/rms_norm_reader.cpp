// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm reader (NCRISC / NoC0).
//
// Per kernel, once:
//   prepare_constants()  — cb_scaler (reduce scaler 1.0), cb_wmask (0/1 column
//                          mask for the ragged hidden tile), cb_zero_tile.
//   load_gamma_slice()   — cb_gamma_tiles: this core's core_w_tiles gamma tiles,
//                          resident for the whole kernel (never popped by
//                          compute), so gamma crosses DRAM once per core.
//
// Per block (a block_row_tiles x core_w_tiles tile rectangle):
//   load_block()         — the whole block behind ONE read barrier on the tiled
//                          path, or one barrier per 32-row block (== core_w_tiles
//                          tiles) via read_sticks_for_tilize on the ROW_MAJOR path.
//
// Helper substitutions (raw NoC instead of a kernel_lib helper), with reasons:
//   * load_block on the TILE path uses raw noc_async_read_tile over a
//     TensorAccessor. read_sticks_for_tilize is ROW-MAJOR ONLY: it derives a
//     stick stride and asserts tile_size % tile_hw == 0
//     (tilize_helpers_dataflow.inl:82-85), and a TILE tensor's DRAM pages are
//     already tiles, so it has no sticks to read. The RM branch below does use
//     the helper.
//   * cb_zero_tile is filled with a direct L1 memset: it is the identity operand
//     of the combine's DEST accumulation, not a reduce scaler, so the
//     reduce-scaler helpers (whose contract is "reduce LLK only") do not apply.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_input_tiles = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_wmask = 3;
constexpr uint32_t cb_zero_tile = 4;
constexpr uint32_t cb_gamma_rm = 12;
constexpr uint32_t cb_gamma_tiles = 13;
constexpr uint32_t TILE_HW_DIM = 32;
}  // namespace

void kernel_main() {
    constexpr uint32_t CORE_W_TILES = get_compile_time_arg_val(0);
    constexpr uint32_t TENSOR_W_TILES = get_compile_time_arg_val(1);
    constexpr bool IS_RM_IN = get_compile_time_arg_val(2) != 0;
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(3) != 0;
    constexpr bool IS_RM_GAMMA = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t PARTIAL_W = get_compile_time_arg_val(5);
    constexpr uint32_t IN_ELEM_BYTES = get_compile_time_arg_val(6);
    constexpr uint32_t GAMMA_ELEM_BYTES = get_compile_time_arg_val(7);
    constexpr auto src_args = TensorAccessorArgs<8>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t row_tile_start = get_arg_val<uint32_t>(2);
    const uint32_t num_blocks = get_arg_val<uint32_t>(3);
    const uint32_t block_row_tiles = get_arg_val<uint32_t>(4);
    const uint32_t last_block_row_tiles = get_arg_val<uint32_t>(5);
    const uint32_t w_tile_start = get_arg_val<uint32_t>(6);
    const uint32_t owns_last_w_tile = get_arg_val<uint32_t>(7);
    const uint32_t num_sticks = get_arg_val<uint32_t>(8);
    const uint32_t stick_start = get_arg_val<uint32_t>(9);
    const uint32_t in_slice_bytes = get_arg_val<uint32_t>(10);
    const uint32_t in_byte_offset = get_arg_val<uint32_t>(11);
    [[maybe_unused]] const uint32_t gamma_slice_bytes = get_arg_val<uint32_t>(12);
    [[maybe_unused]] const uint32_t gamma_byte_offset = get_arg_val<uint32_t>(13);

    // ---------------- prepare_constants (once per kernel) ----------------
    // Pool-type-aware overload: SUM/REDUCE_ROW fills the matmul-path scaler
    // layout. The masking of the ragged hidden tile is done numerically by the
    // compute kernel, so the scaler here is a plain 1.0.
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    if constexpr (PARTIAL_W != 0) {
        if (owns_last_w_tile) {
            // 1.0 in columns [0, PARTIAL_W), 0 elsewhere, in the row-0
            // broadcast layout the compute kernel consumes with BroadcastDim::Row.
            dataflow_kernel_lib::prepare_reduce_mask<cb_wmask, ckernel::ReduceDim::REDUCE_ROW>(PARTIAL_W);
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
            // One zero-padded stick; compute tilizes it into the row-0-valid
            // tile form that TILE gamma already has.
            const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr);
            dataflow_kernel_lib::read_sticks_for_tilize<cb_gamma_rm, dataflow_kernel_lib::TilizeGranularity::TILE>(
                gamma_acc, 1, gamma_slice_bytes, 0, gamma_byte_offset);
        } else {
            const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma_tiles);
            const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr, gamma_tile_bytes);
            cb_reserve_back(cb_gamma_tiles, CORE_W_TILES);
            const uint32_t dst = get_write_ptr(cb_gamma_tiles);
            for (uint32_t c = 0; c < CORE_W_TILES; ++c) {
                noc_async_read_tile(w_tile_start + c, gamma_acc, dst + c * gamma_tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_gamma_tiles, CORE_W_TILES);
        }
    }

    // ---------------- load_block (per block) -----------------------------
    if constexpr (IS_RM_IN) {
        const auto acc = TensorAccessor(src_args, src_addr);
        uint32_t sticks_done = 0;
        for (uint32_t b = 0; b < num_blocks; ++b) {
            const uint32_t rows_t = (b + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles;
            uint32_t sticks_this = rows_t * TILE_HW_DIM;
            if (sticks_this > num_sticks - sticks_done) {
                sticks_this = num_sticks - sticks_done;
            }
            // One barrier per 32-row block == core_w_tiles tiles per barrier.
            dataflow_kernel_lib::read_sticks_for_tilize<cb_input_rm, dataflow_kernel_lib::TilizeGranularity::TILE>(
                acc, sticks_this, in_slice_bytes, stick_start + sticks_done, in_byte_offset);
            sticks_done += sticks_this;
        }
    } else {
        const uint32_t tile_bytes = get_tile_size(cb_input_tiles);
        const auto acc = TensorAccessor(src_args, src_addr, tile_bytes);
        for (uint32_t b = 0; b < num_blocks; ++b) {
            const uint32_t rows_t = (b + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles;
            const uint32_t pages = rows_t * CORE_W_TILES;
            cb_reserve_back(cb_input_tiles, pages);
            const uint32_t dst = get_write_ptr(cb_input_tiles);
            for (uint32_t r = 0; r < rows_t; ++r) {
                const uint32_t row_tile = row_tile_start + b * block_row_tiles + r;
                const uint32_t base = row_tile * TENSOR_W_TILES + w_tile_start;
                for (uint32_t c = 0; c < CORE_W_TILES; ++c) {
                    noc_async_read_tile(base + c, acc, dst + (r * CORE_W_TILES + c) * tile_bytes);
                }
            }
            // The whole block behind one barrier — never one barrier per tile.
            noc_async_read_barrier();
            cb_push_back(cb_input_tiles, pages);
        }
    }
}
