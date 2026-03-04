// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for dit_rms_norm_unary_fused with ROW_MAJOR input (large-tensor path).
//
// The large-tensor compute kernel pops cb_in per block and reads it twice per ncht
// (RMSNORM path): once for the variance loop (TILIZE_IN Pass 1) and once for the
// normalization loop (TILIZE_IN Pass 2).  For non-RMSNORM it reads three times.
// This reader pushes the row-major input data into cb_in_rm for each required pass.
//
// Compile-time args:
//   CTA[0]     = block_size (block size in tiles)
//   CTA[1..]   = TensorAccessorArgs for input a (ROW_MAJOR, page_size = W * elem_size)
//   ...        = TensorAccessorArgs for b / residual (TILE, may be null)
//   ...        = TensorAccessorArgs for gamma (TILE, may be null)
//   ...        = TensorAccessorArgs for beta  (TILE, may be null)
//   CTA[last]  = elem_size_bytes (element size in bytes of the input tensor)
//
// Runtime args:
//   arg[0] = src_addr          (input a base address)
//   arg[1] = NCHt              (number of tile-rows assigned to this core)
//   arg[2] = Wt                (width in tiles)
//   arg[3] = row_offset        (absolute starting row = curr_tile_row * TILE_HEIGHT)
//   arg[4] = packed_one_value  (scaler value for reduce)
//   arg[5] = eps               (epsilon as bit-cast uint32)
//   arg[6] = gamma_dram_addr
//   arg[7] = beta_dram_addr
//   arg[8] = b_dram_addr       (residual, unused if no FUSE_PRE_ADD)
//   arg[9] = W                 (logical width in elements)
//   arg[10] = H_logical        (total valid rows — avoids OOB reads when H % 32 != 0)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "ttnn/operations/normalization/kernel_util/dataflow/custom_tiles.h"
#include "layernorm_dataflow_utils.h"

namespace generic = norm::kernel_util::generic;
namespace layernorm_dataflow_utils = norm::layernorm::device::kernels::dataflow;

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t NCHt = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t row_offset = get_arg_val<uint32_t>(3);  // abs start row = curr_tile_row * TILE_HEIGHT
    const uint32_t gamma_addr = get_arg_val<uint32_t>(6);
    const uint32_t beta_addr = get_arg_val<uint32_t>(7);
    const uint32_t b_addr = get_arg_val<uint32_t>(8);
    const uint32_t W = get_arg_val<uint32_t>(9);
    const uint32_t H_logical = get_arg_val<uint32_t>(10);  // total valid (non-padded) rows

    // constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0; // Unused
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_in_rm = tt::CBIndex::c_27;
    constexpr uint32_t cb_id_gamma = tt::CBIndex::c_5;
    constexpr uint32_t cb_id_beta = tt::CBIndex::c_6;

    // No use_welford arg (large-tensor + welford uses a separate kernel).
    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr auto src0_args = TensorAccessorArgs<1>();
    [[maybe_unused]] constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<src1_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    constexpr uint32_t elem_size_bytes = get_compile_time_arg_val(beta_args.next_compile_time_args_offset());

    // ROW_MAJOR accessor: one page = one full row (W * elem_size bytes).
    // get_noc_addr(row_idx, src_a) returns the NOC address of the start of row row_idx.
    const uint32_t rm_page_size = W * elem_size_bytes;
    const auto src_a = TensorAccessor(src0_args, src_addr, rm_page_size);

#ifdef FUSE_GAMMA
    const uint32_t gamma_tile_bytes = get_tile_size(cb_id_gamma);
    const auto addrg = TensorAccessor(gamma_args, gamma_addr, gamma_tile_bytes);
#endif
#ifdef FUSE_BETA
    const uint32_t beta_tile_bytes = get_tile_size(cb_id_beta);
    const auto addrb = TensorAccessor(beta_args, beta_addr, beta_tile_bytes);
#endif
#ifdef FUSE_PRE_ADD
    const uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    const auto src_b = TensorAccessor(src1_args, b_addr, src1_tile_bytes);
    // b is TILE layout: compute tile_offset from row_offset
    const uint32_t tile_offset = (row_offset / tt::constants::TILE_HEIGHT) * Wt;
#endif

    {
        constexpr uint32_t cb_in_2 = tt::CBIndex::c_2;
        const uint32_t scaler = get_arg_val<uint32_t>(4);
        generate_reduce_scaler(cb_in_2, scaler);
        const auto partial_last_tile_cols = W % tt::constants::TILE_WIDTH;
        if (partial_last_tile_cols > 0) {
            norm::kernel_util::dataflow::generate_partial_reduce_scaler(cb_in_2, scaler, partial_last_tile_cols);
        }
    }
    constexpr uint32_t eps_cb_id = 3;
    const uint32_t eps = get_arg_val<uint32_t>(5);
    generate_bcast_col_scalar(eps_cb_id, eps);

    // cb_in_rm row stride: full-block-width of row-major elements per row.
    // Each row in cb_in_rm occupies block_size * TILE_WIDTH * elem_size bytes.
    constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;
    constexpr uint32_t TILE_W = tt::constants::TILE_WIDTH;
    constexpr uint32_t full_row_stride = block_size * TILE_W * elem_size_bytes;

    uint32_t tile_offs = 0;
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        const uint32_t abs_tile_row = row_offset / TILE_H + ncht;

#ifndef RMSNORM
        // Data for Calculating E[X]
        layernorm_dataflow_utils::push_row_major_blocks_to_cb<decltype(src_a), TILE_W, TILE_H>(
            cb_id_in_rm, src_a, Wt, block_size, abs_tile_row, elem_size_bytes, full_row_stride, H_logical);
#ifdef FUSE_PRE_ADD
        for (auto block : generic::blocks(Wt, block_size)) {
            layernorm_dataflow_utils::read_block_to_cb(
                cb_id_in1, src_b, src1_tile_bytes, tile_offs + block.start() + tile_offset, block);
        }
#endif
#endif
        // Pass 1 for TILIZE_IN: variance calculation
        layernorm_dataflow_utils::push_row_major_blocks_to_cb<decltype(src_a), TILE_W, TILE_H>(
            cb_id_in_rm, src_a, Wt, block_size, abs_tile_row, elem_size_bytes, full_row_stride, H_logical);
#ifdef FUSE_PRE_ADD
        for (auto block : generic::blocks(Wt, block_size)) {
            layernorm_dataflow_utils::read_block_to_cb(
                cb_id_in1, src_b, src1_tile_bytes, tile_offs + block.start() + tile_offset, block);
        }
#endif
        // Pass 2 for TILIZE_IN: normalization — input MUST be interleaved with gamma/beta
        // per block.  Pushing all pass-2 input first and then all gamma deadlocks:
        //   cb_in_rm capacity = 1 block; the compute's normalization loop reads gamma
        //   INSIDE the same per-block loop that drains cb_in_rm.  Once compute finishes
        //   the x/sqrt(var+eps) step it blocks on cb_gamma, while the reader is still
        //   blocked on cb_in_rm for the next block — circular wait.
        // Number of valid rows for this tile-row (same for all blocks within the ncht).
        // TODO: Simplify num_valid_rows_pass2 logic
        const uint32_t abs_row_start = abs_tile_row * TILE_H;
        const uint32_t num_valid_rows_pass2 =
            (abs_row_start >= H_logical) ? 0u
                                         : (H_logical - abs_row_start < TILE_H ? H_logical - abs_row_start : TILE_H);

        for (auto block : generic::blocks(Wt, block_size)) {
            // Pass 2 input for this block
            const uint32_t col_byte_offset = block.start() * TILE_W * elem_size_bytes;
            const uint32_t row_read_bytes = block.size() * TILE_W * elem_size_bytes;
            cb_reserve_back(cb_id_in_rm, block.full_block_size());
            // TODO: Move to separate function
            uint32_t l1_base = get_write_ptr(cb_id_in_rm);

            // Zero-fill padding rows so tilize_block sees 0 instead of stale L1 data.
            // if (num_valid_rows_pass2 < TILE_H) {
            //     volatile tt_l1_ptr uint32_t* p =
            //         reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_base + num_valid_rows_pass2 * full_row_stride);
            //     const uint32_t pad_words = (TILE_H - num_valid_rows_pass2) * full_row_stride / sizeof(uint32_t);
            //     for (uint32_t i = 0; i < pad_words; ++i) {
            //         p[i] = 0;
            //     }
            // }

            uint32_t l1_ptr = l1_base;
            for (uint32_t row = 0; row < num_valid_rows_pass2; ++row) {
                const uint64_t noc_addr = get_noc_addr(abs_tile_row * TILE_H + row, src_a) + col_byte_offset;
                noc_async_read(noc_addr, l1_ptr, row_read_bytes);
                l1_ptr += full_row_stride;
            }
            noc_async_read_barrier();
            cb_push_back(cb_id_in_rm, block.full_block_size());

            // Gamma/beta for this block — pushed immediately after input so compute
            // finds them in the CB when it reaches the gamma/beta multiplication step.
#ifdef FUSE_PRE_ADD
            layernorm_dataflow_utils::read_block_to_cb(
                cb_id_in1, src_b, src1_tile_bytes, tile_offs + block.start() + tile_offset, block);
#endif
#ifdef FUSE_GAMMA
            layernorm_dataflow_utils::read_block_to_cb(cb_id_gamma, addrg, gamma_tile_bytes, block.start(), block);
#endif
#ifdef FUSE_BETA
            layernorm_dataflow_utils::read_block_to_cb(cb_id_beta, addrb, beta_tile_bytes, block.start(), block);
#endif
        }

        tile_offs += Wt;
    }
    DPRINT << "end of reader kernel" << ENDL();
}
