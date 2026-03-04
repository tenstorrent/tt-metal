// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for dit_rms_norm_unary_fused with ROW_MAJOR input (standard / non-large-tensor path).
//
// For each ncht (tile-row) this kernel reads TILE_HEIGHT rows of row-major input data from DRAM,
// one column block at a time, into cb_in_rm (CB 27).  The compute kernel's TILIZE_IN block
// then converts cb_in_rm → cb_in (CB 0) using tilize_block().
//
// Gamma and beta (if present) are always TILE layout and read once at ncht==0.
//
// Compile-time args:
//   CTA[0]     = block_size (block size in tiles)
//   CTA[1]     = use_welford
//   CTA[2..]   = TensorAccessorArgs for input a (ROW_MAJOR, page_size = W * elem_size)
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

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "ttnn/operations/normalization/kernel_util/dataflow/custom_tiles.h"
#include "layernorm_dataflow_utils.h"

namespace generic = norm::kernel_util::generic;
namespace layernorm_dataflow_utils = norm::layernorm::device::kernels::dataflow;

// DEBUG
void print_cb_tile(uint32_t cb, uint32_t tile_idx) {
    volatile uint16_t* ptr = reinterpret_cast<volatile uint16_t*>(get_read_ptr(cb));
    ptr += tile_idx * 32 * 32;

    for (int subtile_i = 0; subtile_i < 2; subtile_i++) {
        // Iterate through 16 rows within each subtile row
        for (int local_row = 0; local_row < 16; local_row++) {
            // Calculate the actual row in original matrix
            int row = subtile_i * 16 + local_row;
            // Iterate through 2x2 subtiles horizontally
            for (int subtile_j = 0; subtile_j < 2; subtile_j++) {
                // Iterate through 16 columns within each subtile
                for (int local_col = 0; local_col < 16; local_col++) {
                    // Calculate the actual column in original matrix
                    int col = subtile_j * 16 + local_col;
                    // Calculate index using only multiplication and addition
                    auto index = local_row * 16 + local_col + subtile_i * 512 + subtile_j * 256;
                    // /*element_offset=*/index);ptr
                    DPRINT << BF16(ptr[index]) << ", ";
                }
            }
            DPRINT << ENDL();
        }
    }  // subtile_i
}

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t NCHt = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t row_offset = get_arg_val<uint32_t>(3);  // abs start row = curr_tile_row * TILE_HEIGHT
    const uint32_t gamma_addr = get_arg_val<uint32_t>(6);
    const uint32_t beta_addr = get_arg_val<uint32_t>(7);
    const uint32_t b_addr = get_arg_val<uint32_t>(8);
    const uint32_t W_logical = get_arg_val<uint32_t>(9);
    const uint32_t H_logical = get_arg_val<uint32_t>(10);  // total valid (non-padded) rows

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_in_rm = tt::CBIndex::c_27;
    constexpr uint32_t cb_id_gamma = tt::CBIndex::c_5;
    constexpr uint32_t cb_id_beta = tt::CBIndex::c_6;

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr bool use_welford = get_compile_time_arg_val(1) == 1;
    constexpr auto src0_args = TensorAccessorArgs<2>();
    [[maybe_unused]] constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<src1_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    constexpr uint32_t elem_size_bytes = get_compile_time_arg_val(beta_args.next_compile_time_args_offset());

    // ROW_MAJOR accessor: one page = one full row (W * elem_size bytes).
    // get_noc_addr(row_idx, src_a) returns the NOC address of the start of row row_idx.
    const uint32_t rm_page_size = W_logical * elem_size_bytes;
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

    // Generate constant tiles (scaler and epsilon)
    if constexpr (!use_welford) {
        constexpr uint32_t cb_in_2 = tt::CBIndex::c_2;
        const uint32_t scaler = get_arg_val<uint32_t>(4);
        generate_reduce_scaler(cb_in_2, scaler);
        const auto partial_last_tile_cols = W_logical % tt::constants::TILE_WIDTH;
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

    uint32_t tile_offs = 0;  // running tile offset for FUSE_PRE_ADD b-tensor reads
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // TODO: Remove division
        const uint32_t abs_tile_row = row_offset / TILE_H + ncht;

        // Push one column block of ROW_MAJOR input into cb_in_rm per iteration.
        // The compute kernel tilizes each block before the existing computation.
        layernorm_dataflow_utils::push_row_major_blocks_to_cb<decltype(src_a), TILE_W, TILE_H>(
            cb_id_in_rm, src_a, Wt, block_size, abs_tile_row, elem_size_bytes, full_row_stride, H_logical);

        print_cb_tile(cb_id_in_rm, 7);

#ifdef FUSE_PRE_ADD
        for (auto block : generic::blocks(Wt, block_size)) {
            layernorm_dataflow_utils::read_block_to_cb(
                cb_id_in1, src_b, src1_tile_bytes, tile_offs + block.start() + tile_offset, block);
        }
#endif

#if defined FUSE_GAMMA || defined FUSE_BETA
        if (ncht == 0) {
            for (auto block : generic::blocks(Wt, block_size)) {
#ifdef FUSE_GAMMA
                layernorm_dataflow_utils::read_block_to_cb(cb_id_gamma, addrg, gamma_tile_bytes, block.start(), block);
#endif
#ifdef FUSE_BETA
                layernorm_dataflow_utils::read_block_to_cb(cb_id_beta, addrb, beta_tile_bytes, block.start(), block);
#endif
            }
        }
#endif
        tile_offs += Wt;
    }
}
