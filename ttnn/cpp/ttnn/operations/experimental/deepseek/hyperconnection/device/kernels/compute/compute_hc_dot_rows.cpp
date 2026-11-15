// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/experimental/add_rsqrt.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/experimental/pack_block.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "hc_rows_sfpu.h"

// The fn dot of the single-user hyperconnection program for a block of ROWS rows (ROWS in {2, 4, 8}): the
// compute_large_k_tree_reduce flow (matmul_decode) with a (ROWS x 32) in0 tile, so the whole block is one custom_mm
// pass. Every partial, the tree's running sums and fused_w are single (ROWS x 32) tiles.
//
// With FUSE_RMS_NORM each partial is followed by a second (ROWS x 32) tile whose row j holds this core's sum(A_j^2)
// in every column (hc_dot_rows_sumsq), summed up the tree along with the partial. The root then scales its result
// elementwise by rsqrt(sum / K + eps), all on (ROWS x 32) tiles.
using namespace ckernel;
void kernel_main() {
    constexpr uint32_t Kc_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t rows = get_compile_time_arg_val(1);
    constexpr uint32_t dst_num_tiles = get_compile_time_arg_val(2);

    const uint32_t num_children = get_arg_val<uint32_t>(0);
    const uint32_t is_root = get_arg_val<uint32_t>(1);

    constexpr uint32_t in0_cb_id = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t in1_cb_id = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t partial_cb_id = get_named_compile_time_arg_val("cb_partial");
    constexpr uint32_t recv_cb_id = get_named_compile_time_arg_val("cb_recv");
    constexpr uint32_t out_cb_id = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t send_cb_id = get_named_compile_time_arg_val("cb_send");

#ifdef FUSE_RMS_NORM
    constexpr uint32_t rms_inv_k_bits = get_compile_time_arg_val(3);
    constexpr uint32_t rms_epsilon_bits = get_compile_time_arg_val(4);
    constexpr uint32_t block_num_tiles = 2;
    const uint32_t result_cb_id = send_cb_id;
    static_assert(Kc_tiles <= 8, "the per-row sum(A^2) holds the K-slice's tiles in a half-sync 16-bit Dst");
#else
    constexpr uint32_t block_num_tiles = 1;
    const uint32_t result_cb_id = is_root ? out_cb_id : send_cb_id;
#endif
    const uint32_t matmul_cb_id = num_children > 0 ? partial_cb_id : result_cb_id;

    CircularBuffer in0_cb(in0_cb_id);
    CircularBuffer in1_cb(in1_cb_id);
    CircularBuffer matmul_cb(matmul_cb_id);

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb_id, in1_cb_id, matmul_cb_id);

    in0_cb.wait_front(Kc_tiles);
    in1_cb.wait_front(Kc_tiles);
    matmul_cb.reserve_back(block_num_tiles);

    constexpr auto layout = CustomMmDestLayout::DenseSplitAcc;
    static_assert(Kc_tiles >= 2 && Kc_tiles <= 256 && Kc_tiles % 2 == 0);
    custom_mm_block_init_short<false, layout>(in0_cb_id, in1_cb_id, matmul_cb_id, 1);
    pack_block_contiguous_init(matmul_cb_id);
    tile_regs_acquire();
    custom_mm_block<layout, true>(in0_cb_id, in1_cb_id, 0, 0, 0, Kc_tiles, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_block_contiguous(0, matmul_cb_id, 1);
    tile_regs_release();
    custom_mm_block_uninit<layout>(matmul_cb_id);

#ifdef FUSE_RMS_NORM
    // custom_mm_block_uninit leaves the packer on a 32x32 tile stride; reprogram for the (rows x 32) tiles.
    compute_kernel_hw_startup(in0_cb_id, in0_cb_id, matmul_cb_id);
    tile_regs_acquire();
    copy_init(in0_cb_id);
    for (uint32_t t = 0; t < Kc_tiles; ++t) {
        copy_tile(in0_cb_id, t, t);
    }
    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>()));
    MATH((ckernel::sfpu::hc_dot_rows_sumsq<rows, Kc_tiles>()));
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, matmul_cb_id, 1);
    tile_regs_release();
#endif
    matmul_cb.push_back(block_num_tiles);
    in0_cb.pop_front(Kc_tiles);
    in1_cb.pop_front(Kc_tiles);

    if (num_children > 0) {
        CircularBuffer partial_cb(partial_cb_id);
        CircularBuffer recv_cb(recv_cb_id);
        CircularBuffer result_cb(result_cb_id);

        compute_kernel_hw_startup(partial_cb_id, recv_cb_id, result_cb_id);

        partial_cb.wait_front(block_num_tiles);
        result_cb.reserve_back(block_num_tiles);
        tile_regs_acquire();
        copy_init(partial_cb_id);
        for (uint32_t t = 0; t < block_num_tiles; ++t) {
            copy_tile(partial_cb_id, t, t);
        }
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(recv_cb_id);
        for (uint32_t child = 0; child < num_children; ++child) {
            recv_cb.wait_front((child + 1) * block_num_tiles);
            for (uint32_t t = 0; t < block_num_tiles; ++t) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                    recv_cb_id, child * block_num_tiles + t, t);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < block_num_tiles; ++t) {
            pack_tile<true>(t, result_cb_id, t);
        }
        tile_regs_release();
        result_cb.push_back(block_num_tiles);
        partial_cb.pop_front(block_num_tiles);
        recv_cb.pop_front(num_children * block_num_tiles);
    }

#ifdef FUSE_RMS_NORM
    if (!is_root) {
        return;
    }
    CircularBuffer sum_cb(send_cb_id);
    CircularBuffer out_cb(out_cb_id);
    compute_kernel_hw_startup(send_cb_id, send_cb_id, out_cb_id);
    sum_cb.wait_front(block_num_tiles);
    out_cb.reserve_back(1);
    tile_regs_acquire();
    copy_init(send_cb_id);
    copy_tile(send_cb_id, 0, 0);
    copy_tile(send_cb_id, 1, 1);
    // Every column of row j of tile 1 holds sum(A_j^2): the scale, then the row times it, are elementwise.
    add_rsqrt_tile_init();
    add_rsqrt_tile<false, VectorMode::RC, 8, false, rms_inv_k_bits>(1, rms_epsilon_bits);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_cb_id);
    tile_regs_release();
    out_cb.push_back(1);
    sum_cb.pop_front(block_num_tiles);
#endif
}
