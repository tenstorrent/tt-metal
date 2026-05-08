// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
// Dense RM W path: chunk packed Ht and Wt; one tilize pass per W chunk (all H slabs), then one reduce()
// per W chunk with ReduceInputBlockShape::of(ht_in_chunk, wt_in_chunk, NC). Host passes ht/wt chunk sizes.
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

#ifdef REDUCE_POST_MUL
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

void kernel_main() {
    const uint32_t Ht = get_compile_time_arg_val(0);
    const uint32_t Wt = get_compile_time_arg_val(1);
    const uint32_t NC = get_compile_time_arg_val(2);
    const uint32_t wt_tiles_per_chunk = get_compile_time_arg_val(3);
    const uint32_t ht_tiles_per_chunk = get_compile_time_arg_val(4);

    constexpr uint32_t cb_rm = tt::CBIndex::c_24;
    constexpr uint32_t cb_tile_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_3;
    constexpr uint32_t cb_acc = tt::CBIndex::c_5;

    compute_kernel_hw_startup(cb_rm, cb_tile_in);

    for (uint32_t ht_base = 0; ht_base < Ht; ht_base += ht_tiles_per_chunk) {
        const uint32_t ht_in_chunk = (ht_base + ht_tiles_per_chunk < Ht) ? ht_tiles_per_chunk : (Ht - ht_base);
        uint32_t chunk_idx = 0;
        for (uint32_t wt_base = 0; wt_base < Wt; wt_base += wt_tiles_per_chunk) {
            const uint32_t wt_in_chunk = (wt_base + wt_tiles_per_chunk < Wt) ? wt_tiles_per_chunk : (Wt - wt_base);
            const bool is_last_chunk = (wt_base + wt_in_chunk) == Wt;

            tilize_init(cb_rm, wt_in_chunk, cb_tile_in);

            for (uint32_t hti = 0; hti < ht_in_chunk; ++hti) {
                cb_wait_front(cb_rm, 1);
                cb_reserve_back(cb_tile_in, wt_in_chunk);
                tilize_block(cb_rm, wt_in_chunk, cb_tile_in);
                cb_pop_front(cb_rm, 1);
                cb_push_back(cb_tile_in, wt_in_chunk);
            }

            tilize_uninit(cb_rm, cb_tile_in);

            if (is_last_chunk) {
                compute_kernel_lib::reduce<
                    REDUCE_OP,
                    REDUCE_DIM,
                    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
                    compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
                    cb_tile_in,
                    cb_scaler,
                    cb_out,
                    compute_kernel_lib::ReduceInputBlockShape::of(ht_in_chunk, wt_in_chunk, NC),
                    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                    compute_kernel_lib::Accumulate::at(cb_acc, chunk_idx),
#ifdef REDUCE_POST_MUL
                    [](uint32_t dst_idx) {
                        constexpr uint32_t post_mul_scaler_bits = get_compile_time_arg_val(5);
                        binop_with_scalar_tile_init();
                        mul_unary_tile(dst_idx, post_mul_scaler_bits);
                    }
#else
                    compute_kernel_lib::NoOp{}
#endif
                );
            } else {
                compute_kernel_lib::reduce<
                    REDUCE_OP,
                    REDUCE_DIM,
                    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
                    compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
                    cb_tile_in,
                    cb_scaler,
                    cb_acc,
                    compute_kernel_lib::ReduceInputBlockShape::of(ht_in_chunk, wt_in_chunk, NC),
                    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                    compute_kernel_lib::Accumulate::at(cb_acc, chunk_idx),
                    compute_kernel_lib::NoOp{});
            }
            ++chunk_idx;
        }
    }
}
