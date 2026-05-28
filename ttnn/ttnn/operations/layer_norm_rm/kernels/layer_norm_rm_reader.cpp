// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// layer_norm_rm reader — runs on NCRISC.
//
// Per-core work:
//   1. Boot one-shot: push the scaler tile(s) into cb_scaler using a
//      pool-type/reduce-dim-aware overload of the reduce-scaler API.
//        - tile-aligned W (partial_w == 0): a single tile via
//          prepare_reduce_scaler<cb_scaler, SUM, REDUCE_ROW>(1/W).
//        - W non-aligned (partial_w > 0, Refinement 3): a (full, partial) pair
//          via prepare_partial_reduce_scalers<…, partial_w>(1/W). Compute side
//          uses ReducePartialScaler::last_tile_at(1) to pick the partial tile
//          for the last reduce-dim iteration of accumulate_reduce_block.
//      The compute kernel never pops these scalers until end-of-kernel.
//   2. Strip loop: for each of `num_strips` strips, stream the strip three
//      times through cb_input_rm (Pass A → Pass B → Pass C).
//      During Pass C, also stream the gamma and beta chunks (when present)
//      through cb_gamma_rm / cb_beta_rm.
//
// Per-strip mapping:
//   strip s spans 32 contiguous RM rows starting at row (start_strip + s) * 32.
//   For each chunk c in [0, NUM_BLOCKS), the chunk covers W positions
//   [c * BLOCK_SIZE * 32, (c+1) * BLOCK_SIZE * 32) of those rows.
//
// Refinement 3 — non-tile-aligned shapes:
//   * W non-aligned: the LAST chunk's `row_bytes` passed to
//     read_sticks_for_tilize is `input_chunk_bytes_last` (and similarly
//     `affine_chunk_bytes_last` for gamma/beta), which equals the actual valid
//     byte count for the chunk's W coverage instead of `BLOCK_SIZE*32*bpe`.
//     The reader helper pads the L1 stride internally so the CB still receives
//     `width_in_tiles=BLOCK_SIZE` tile-pages per chunk; the padded positions
//     hold whatever leftover L1 data was there. The compute side masks those
//     positions via the partial-scaler tile on the reduce, and the writer
//     drops them by writing only `output_chunk_bytes_last` bytes per row.
//   * H non-aligned: when the strip's global index equals last_strip_idx, the
//     helper is called with `total_num_rows = last_strip_rows < 32` instead of
//     32. The TILE-granularity read helper handles partial-row blocks
//     internally (it still pushes `width_in_tiles` tile-pages — the unwritten
//     L1 rows hold leftover data — and compute sees full 32-row tiles whose
//     padded rows are never written back to DRAM).
//
// CB granularity:
//   cb_input_rm     : TILE granularity (page_size = tile_size).
//   cb_gamma_rm/beta: ROW granularity  (page_size = chunk_bytes).
//
// CT arg layout (see program descriptor):
//   [0]  BLOCK_SIZE
//   [1]  NUM_BLOCKS
//   [2]  input_chunk_bytes       (BLOCK_SIZE * 32 * input_bpe)
//   [3]  affine_chunk_bytes      (BLOCK_SIZE * 32 * affine_bpe)
//                                differs from input_chunk_bytes when input dtype != affine dtype
//   [4]  scaler_bits             (= __builtin_bit_cast(uint32_t, 1.0f / W))
//   [5]  HAS_GAMMA
//   [6]  HAS_BETA
//   [7]  partial_w               Refinement 3 — # of valid W cols in the LAST tile (0 ⇒ aligned)
//   [8]  input_chunk_bytes_last  Refinement 3 — actual valid bytes per row for the last chunk
//   [9]  affine_chunk_bytes_last Refinement 3 — same for gamma/beta
//   [10] last_strip_idx          Refinement 3 — index of the global last strip
//   [11] last_strip_rows         Refinement 3 — # of valid rows in the global last strip (∈ [1,32])
//   [12..] TensorAccessorArgs(input)
//   [..]   TensorAccessorArgs(gamma) — placeholder when absent
//   [..]   TensorAccessorArgs(beta)  — placeholder when absent
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
    constexpr uint32_t input_chunk_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t affine_chunk_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(4);
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(5);
    constexpr uint32_t HAS_BETA = get_compile_time_arg_val(6);
    constexpr uint32_t partial_w = get_compile_time_arg_val(7);
    constexpr uint32_t input_chunk_bytes_last = get_compile_time_arg_val(8);
    constexpr uint32_t affine_chunk_bytes_last = get_compile_time_arg_val(9);
    constexpr uint32_t last_strip_idx = get_compile_time_arg_val(10);
    constexpr uint32_t last_strip_rows = get_compile_time_arg_val(11);
    (void)BLOCK_SIZE;

    // CT args for accessors are chained — declared unconditionally so the
    // CT-arg offsets line up regardless of which optional tensors are
    // present. The actual TensorAccessor instantiation is gated by the
    // HAS_GAMMA / HAS_BETA flags below.
    constexpr auto input_args = TensorAccessorArgs<12>();
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
    //
    // Refinement 3: when W is non-aligned (partial_w > 0), emit a (full,
    // partial) tile pair. The partial tile has col-0 filled only on rows
    // [0, partial_w) — rows >= partial_w hold 0, masking the padded W
    // positions out of the SUM reduce. Compute picks tile 1 (partial) for the
    // last reduce-dim iteration via ReducePartialScaler::last_tile_at(1).
    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    if constexpr (partial_w > 0) {
        dataflow_kernel_lib::prepare_partial_reduce_scalers<
            cb_scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            partial_w>(scaler_f);
    } else {
        dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            scaler_f);
    }

    // ── Accessors — built once per kernel run. ──
    // TensorAccessor only carries the address + (constexpr) args, so this
    // is essentially zero-cost; the previous implementation rebuilt the
    // gamma/beta accessors on every chunk inside the strip loop.
    const auto input_accessor = TensorAccessor(input_args, input_addr);

    // ── Strip loop ──
    for (uint32_t i = 0; i < num_strips; ++i) {
        const uint32_t strip = start_strip + i;
        const uint32_t strip_start_row = strip * 32;

        // Refinement 3 — H non-aligned: the global last strip has fewer than
        // 32 valid rows. The read helper handles partial-row blocks natively
        // (it still pushes BLOCK_SIZE tile-pages — unwritten L1 rows hold
        // leftover data — and the writer drops the padded rows on the way
        // out). For tile-aligned shapes this branch is dead (last_strip_rows
        // is 32 and start_strip + i never reaches last_strip_idx with a
        // smaller value).
        const uint32_t rows_this_strip = (strip == last_strip_idx) ? last_strip_rows : 32u;

        // Pass A: read the strip once for mean computation.
        for (uint32_t c = 0; c < NUM_BLOCKS; ++c) {
            // Refinement 3 — W non-aligned: the LAST chunk's actual bytes are
            // less than a full chunk (the chunk's tile-padded W extends past
            // the logical W). Use input_chunk_bytes_last for the last chunk.
            // The read helper pads the L1 stride to width_in_tiles*tile_row_bytes
            // internally, so the CB still receives BLOCK_SIZE tile-pages.
            const uint32_t chunk_bytes_this = (c == NUM_BLOCKS - 1) ? input_chunk_bytes_last : input_chunk_bytes;
            dataflow_kernel_lib::read_sticks_for_tilize<cb_input_rm, dataflow_kernel_lib::TilizeGranularity::TILE>(
                input_accessor,
                /*total_num_rows=*/rows_this_strip,
                /*row_bytes=*/chunk_bytes_this,
                /*start_page=*/strip_start_row,
                /*byte_offset_within_page=*/c * input_chunk_bytes);
        }

        // Pass B: read the strip again for variance computation.
        for (uint32_t c = 0; c < NUM_BLOCKS; ++c) {
            const uint32_t chunk_bytes_this = (c == NUM_BLOCKS - 1) ? input_chunk_bytes_last : input_chunk_bytes;
            dataflow_kernel_lib::read_sticks_for_tilize<cb_input_rm, dataflow_kernel_lib::TilizeGranularity::TILE>(
                input_accessor, rows_this_strip, chunk_bytes_this, strip_start_row, c * input_chunk_bytes);
        }

        // Pass C: read the strip once more, optionally gamma and beta.
        // Optional-tensor accessors are constructed inside the if-constexpr
        // branches (scoped to the strip; lifted out of the per-chunk loop).
        // Affine reads use affine_chunk_bytes (may differ from input_chunk_bytes
        // when input dtype != affine dtype — bf16 input + fp32 gamma, etc.).
        for (uint32_t c = 0; c < NUM_BLOCKS; ++c) {
            const uint32_t input_bytes_this = (c == NUM_BLOCKS - 1) ? input_chunk_bytes_last : input_chunk_bytes;
            dataflow_kernel_lib::read_sticks_for_tilize<cb_input_rm, dataflow_kernel_lib::TilizeGranularity::TILE>(
                input_accessor, rows_this_strip, input_bytes_this, strip_start_row, c * input_chunk_bytes);

            // For gamma/beta the byte counts mirror input but scaled by
            // affine_bpe instead of input_bpe (mixed-precision support).
            const uint32_t affine_bytes_this = (c == NUM_BLOCKS - 1) ? affine_chunk_bytes_last : affine_chunk_bytes;

            if constexpr (HAS_GAMMA != 0) {
                const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr);
                dataflow_kernel_lib::read_sticks_for_tilize<cb_gamma_rm, dataflow_kernel_lib::TilizeGranularity::ROW>(
                    gamma_accessor,
                    /*total_num_rows=*/1,
                    /*row_bytes=*/affine_bytes_this,
                    /*start_page=*/0,
                    /*byte_offset_within_page=*/c * affine_chunk_bytes);
            }

            if constexpr (HAS_BETA != 0) {
                const auto beta_accessor = TensorAccessor(beta_args, beta_addr);
                dataflow_kernel_lib::read_sticks_for_tilize<cb_beta_rm, dataflow_kernel_lib::TilizeGranularity::ROW>(
                    beta_accessor, 1, affine_bytes_this, 0, c * affine_chunk_bytes);
            }
        }
    }
}
