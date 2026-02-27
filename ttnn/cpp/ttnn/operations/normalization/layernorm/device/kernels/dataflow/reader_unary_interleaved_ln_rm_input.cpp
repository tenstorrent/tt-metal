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
//   CTA[0]     = blk (block size in tiles)
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

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t NCHt = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t row_offset = get_arg_val<uint32_t>(3);  // abs start row = curr_tile_row * TILE_HEIGHT
    const uint32_t gamma_addr = get_arg_val<uint32_t>(6);
    const uint32_t beta_addr = get_arg_val<uint32_t>(7);
    const uint32_t b_addr = get_arg_val<uint32_t>(8);
    const uint32_t W = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_in_rm = tt::CBIndex::c_27;
    constexpr uint32_t cb_id_gamma = tt::CBIndex::c_5;
    constexpr uint32_t cb_id_beta = tt::CBIndex::c_6;

    constexpr uint32_t blk = get_compile_time_arg_val(0);
    constexpr bool use_welford = get_compile_time_arg_val(1) == 1;
    constexpr auto src0_args = TensorAccessorArgs<2>();
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

    // Generate constant tiles (scaler and epsilon)
    if constexpr (!use_welford) {
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
    // Each row in cb_in_rm occupies blk * TILE_WIDTH * elem_size bytes.
    constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;
    constexpr uint32_t TILE_W = tt::constants::TILE_WIDTH;
    constexpr uint32_t full_row_stride = blk * TILE_W * elem_size_bytes;

    uint32_t tile_offs = 0;  // running tile offset for FUSE_PRE_ADD b-tensor reads
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        const uint32_t abs_tile_row = row_offset / TILE_H + ncht;

        // Push one column block of ROW_MAJOR input into cb_in_rm per iteration.
        // The compute kernel tilizes each block before the existing computation.
        for (auto block : generic::blocks(Wt, blk)) {
            const uint32_t col_byte_offset = block.start() * TILE_W * elem_size_bytes;
            const uint32_t row_read_bytes = block.size() * TILE_W * elem_size_bytes;

            cb_reserve_back(cb_id_in_rm, block.full_block_size());
            const uint32_t l1_base = get_write_ptr(cb_id_in_rm);
            uint32_t l1_ptr = l1_base;

            for (uint32_t row = 0; row < TILE_H; ++row) {
                const uint64_t noc_addr = get_noc_addr(abs_tile_row * TILE_H + row, src_a) + col_byte_offset;
                noc_async_read(noc_addr, l1_ptr, row_read_bytes);
                l1_ptr += full_row_stride;  // advance by full blk-wide row slot
            }
            noc_async_read_barrier();

            // DPRINT: show what was read for first block of first ncht, to diagnose data corruption.
            // Enable by running with TT_METAL_DPRINT_CORES=0,0
            if (ncht == 0 && block.is_first()) {
                DPRINT << "[rm_reader] src_addr=" << src_addr << " abs_tile_row=" << abs_tile_row
                       << " col_byte_off=" << col_byte_offset << " row_read_bytes=" << row_read_bytes
                       << " full_row_stride=" << full_row_stride << " l1_base=" << l1_base << ENDL();
                // Print first 4 BF16 values (first 2 rows, 2 elements each)
                volatile tt_l1_ptr uint16_t* d = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_base);
                DPRINT << "[rm_reader] row0[0]=" << BF16(d[0]) << " row0[1]=" << BF16(d[1])
                       << " row1[0]=" << BF16(d[full_row_stride / elem_size_bytes])
                       << " row1[1]=" << BF16(d[full_row_stride / elem_size_bytes + 1]) << ENDL();
            }

            cb_push_back(cb_id_in_rm, block.full_block_size());
        }

#ifdef FUSE_PRE_ADD
        for (auto block : generic::blocks(Wt, blk)) {
            layernorm_dataflow_utils::read_block_to_cb(
                cb_id_in1, src_b, src1_tile_bytes, tile_offs + block.start() + tile_offset, block);
        }
#endif

#if defined FUSE_GAMMA || defined FUSE_BETA
        if (ncht == 0) {
            for (auto block : generic::blocks(Wt, blk)) {
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
