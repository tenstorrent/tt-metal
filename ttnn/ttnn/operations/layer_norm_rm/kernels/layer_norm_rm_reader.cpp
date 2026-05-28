// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// layer_norm_rm reader — runs on NCRISC.
//
// Per-core work:
//   1. Boot one-shot: push a single 1/W scaler tile into cb_scaler using the
//      pool-type/reduce-dim-aware overload prepare_reduce_scaler<cb_scaler,
//      SUM, REDUCE_ROW>. The compute kernel never pops this tile.
//   2. Strip loop: for each of `num_strips` strips, stream the strip three
//      times through cb_input_rm (Pass A → Pass B → Pass C).
//      During Pass C, also stream the gamma and beta chunks (when present)
//      through cb_gamma_rm / cb_beta_rm.
//
// Per-strip mapping:
//   strip s spans 32 contiguous RM rows starting at row (start_strip + s) * 32.
//   For each chunk c in [0, NUM_BLOCKS), the chunk covers columns
//   [c * chunk_bytes, (c+1) * chunk_bytes) of those 32 rows.
//
// CB granularity:
//   cb_input_rm     : TILE granularity (page_size = tile_size).
//   cb_gamma_rm/beta: ROW granularity  (page_size = chunk_bytes).
//
// CT arg layout (see program descriptor):
//   [0] BLOCK_SIZE
//   [1] NUM_BLOCKS
//   [2] chunk_bytes
//   [3] scaler_bits (= __builtin_bit_cast(uint32_t, 1.0f / W))
//   [4] HAS_GAMMA
//   [5] HAS_BETA
//   [6..] TensorAccessorArgs(input)
//   [..]  TensorAccessorArgs(gamma) — placeholder when absent
//   [..]  TensorAccessorArgs(beta)  — placeholder when absent
//
// RT arg layout:
//   [0] input_addr
//   [1] gamma_addr   (unused when HAS_GAMMA==0)
//   [2] beta_addr    (unused when HAS_BETA==0)
//   [3] num_strips_for_core
//   [4] start_strip_id

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
    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(0);
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(3);
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(4);
    constexpr uint32_t HAS_BETA = get_compile_time_arg_val(5);
    (void)BLOCK_SIZE;

    // CT args for accessors are chained — declared unconditionally so the
    // CT-arg offsets line up regardless of which optional tensors are
    // present. The actual TensorAccessor instantiation is gated by the
    // HAS_GAMMA / HAS_BETA flags below.
    constexpr auto input_args = TensorAccessorArgs<6>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    [[maybe_unused]] const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_strips = get_arg_val<uint32_t>(3);
    const uint32_t start_strip = get_arg_val<uint32_t>(4);

    // ── One-shot scaler push (1/W into cb_scaler, fp32). ──
    // The pool-type-aware overload selects the matmul-path scaler tile layout
    // for (SUM, REDUCE_ROW), which is the layout compute_kernel_lib::reduce
    // uses internally on this combo.
    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        scaler_f);

    // ── Input accessor (RM sticks, one stick per row). ──
    const auto input_accessor = TensorAccessor(input_args, input_addr);

    // ── Strip loop ──
    for (uint32_t i = 0; i < num_strips; ++i) {
        const uint32_t strip = start_strip + i;
        const uint32_t strip_start_row = strip * 32;

        // Pass A: read the strip once for mean computation.
        for (uint32_t c = 0; c < NUM_BLOCKS; ++c) {
            dataflow_kernel_lib::read_sticks_for_tilize<cb_input_rm, dataflow_kernel_lib::TilizeGranularity::TILE>(
                input_accessor,
                /*total_num_rows=*/32,
                /*row_bytes=*/chunk_bytes,
                /*start_page=*/strip_start_row,
                /*byte_offset_within_page=*/c * chunk_bytes);
        }

        // Pass B: read the strip again for variance computation.
        for (uint32_t c = 0; c < NUM_BLOCKS; ++c) {
            dataflow_kernel_lib::read_sticks_for_tilize<cb_input_rm, dataflow_kernel_lib::TilizeGranularity::TILE>(
                input_accessor, 32, chunk_bytes, strip_start_row, c * chunk_bytes);
        }

        // Pass C: read the strip once more, optionally gamma and beta.
        for (uint32_t c = 0; c < NUM_BLOCKS; ++c) {
            dataflow_kernel_lib::read_sticks_for_tilize<cb_input_rm, dataflow_kernel_lib::TilizeGranularity::TILE>(
                input_accessor, 32, chunk_bytes, strip_start_row, c * chunk_bytes);

            if constexpr (HAS_GAMMA != 0) {
                const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr);
                dataflow_kernel_lib::read_sticks_for_tilize<cb_gamma_rm, dataflow_kernel_lib::TilizeGranularity::ROW>(
                    gamma_accessor,
                    /*total_num_rows=*/1,
                    /*row_bytes=*/chunk_bytes,
                    /*start_page=*/0,
                    /*byte_offset_within_page=*/c * chunk_bytes);
            }

            if constexpr (HAS_BETA != 0) {
                const auto beta_accessor = TensorAccessor(beta_args, beta_addr);
                dataflow_kernel_lib::read_sticks_for_tilize<cb_beta_rm, dataflow_kernel_lib::TilizeGranularity::ROW>(
                    beta_accessor, 1, chunk_bytes, 0, c * chunk_bytes);
            }
        }
    }
}
