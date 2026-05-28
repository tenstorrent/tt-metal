// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// layer_norm_rm reader (NCRISC) — per op_design.md.
//
// Boot:
//   1. Push one fp32 scaler tile = 1/W into cb_scaler. The reduce LLK uses
//      scaler as SrcB; the helper `prepare_reduce_scaler<SUM, REDUCE_ROW>`
//      writes the correct col-0-fill layout for the matmul-path SUM+ROW
//      reduce. The same scaler tile drives BOTH the mean reduce (Phase 2)
//      and the variance reduce (Phase 5).
//   2. If has_gamma: read the single gamma stick from DRAM, replicate it
//      into 32 contiguous L1 rows of cb_gamma_rm, push Wt tile-pages. This
//      lets the compute side run the standard `tilize<Wt>(1)` helper.
//   3. Same for beta if has_beta.
//
// Per work-item loop:
//   read_sticks_for_tilize<cb_input_rm, TilizeGranularity::TILE>(
//       input_accessor, /*total_num_rows=*/32, row_bytes,
//       /*start_page=*/(start_tile_row + i) * 32)
//   pushes Wt tile-pages per work-item (one tile-row's worth).
//
// Helpers considered and rejected for the gamma/beta replicate-32× path:
//   read_sticks_for_tilize reads `total_num_rows` *distinct* sticks
//   (start_page + row_idx) — it has no "broadcast 1 source stick to N rows"
//   mode. The minimal raw expansion is the only option.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_gamma_rm = 1;
constexpr uint32_t cb_beta_rm = 2;
constexpr uint32_t cb_scaler = 8;
}  // namespace

void kernel_main() {
    constexpr uint32_t row_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(3);
    constexpr uint32_t has_beta = get_compile_time_arg_val(4);
    constexpr auto input_args = TensorAccessorArgs<5>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(3);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(4);

    // -------------------------------------------------------------------
    // Boot: scaler tile (1/W) for both reductions.
    //
    // Pool-type/reduce-dim-aware overload — SUM + REDUCE_ROW picks the
    // matmul-path col-0 fill (reduce_helpers_dataflow.hpp:46-48). Caller-
    // provided value (1/W) — we cannot use calculate_and_prepare_reduce_scaler
    // because it forces scaler = 1.0 for SUM.
    // -------------------------------------------------------------------
    const float inv_W = 1.0f / static_cast<float>(W);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        inv_W);

    // -------------------------------------------------------------------
    // Boot: gamma replicate-32×.
    //
    // The gamma tensor has logical shape (..., 1, W) — exactly one stick of
    // row_bytes bytes. To feed the standard tilize<Wt>(1) helper, replicate
    // that single stick across 32 L1 rows. After this push, the compute
    // boot does tilize → cb_gamma_tiles and then `cb_wait_front(Wt)` once
    // so subsequent in-place mul calls can use NoWaitNoPop semantics.
    //
    // padded_row_bytes == row_bytes because Phase 0 requires W % 32 == 0.
    // -------------------------------------------------------------------
    if constexpr (has_gamma) {
        const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, row_bytes);
        cb_reserve_back(cb_gamma_rm, Wt);
        uint32_t l1_addr = get_write_ptr(cb_gamma_rm);
        const uint64_t src_noc_addr = gamma_accessor.get_noc_addr(0);
        for (uint32_t r = 0; r < 32; ++r) {
            noc_async_read(src_noc_addr, l1_addr, row_bytes);
            l1_addr += row_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma_rm, Wt);
    }

    if constexpr (has_beta) {
        const auto beta_accessor = TensorAccessor(beta_args, beta_addr, row_bytes);
        cb_reserve_back(cb_beta_rm, Wt);
        uint32_t l1_addr = get_write_ptr(cb_beta_rm);
        const uint64_t src_noc_addr = beta_accessor.get_noc_addr(0);
        for (uint32_t r = 0; r < 32; ++r) {
            noc_async_read(src_noc_addr, l1_addr, row_bytes);
            l1_addr += row_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_beta_rm, Wt);
    }

    // -------------------------------------------------------------------
    // Per-work-item loop: stream the 32-row tile-row into cb_input_rm.
    //
    // Stick layout (RM, interleaved DRAM): page = stick, page_size = W*4.
    // For the i-th tile-row owned by this core, the 32 sticks span pages
    // [start_tile_row + i, start_tile_row + i + 1) * 32.
    // -------------------------------------------------------------------
    const auto input_accessor = TensorAccessor(input_args, input_addr);

    for (uint32_t i = 0; i < num_tile_rows; ++i) {
        const uint32_t start_page = (start_tile_row + i) * 32u;
        dataflow_kernel_lib::read_sticks_for_tilize<cb_input_rm, dataflow_kernel_lib::TilizeGranularity::TILE>(
            input_accessor, /*total_num_rows=*/32u, row_bytes, start_page);
    }
}
