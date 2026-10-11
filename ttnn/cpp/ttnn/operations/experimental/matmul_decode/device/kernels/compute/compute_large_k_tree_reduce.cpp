// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#ifdef USE_CUSTOM_MM
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/experimental/pack_block.h"
#endif
#ifdef FUSE_RMS_NORM
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/experimental/add_rsqrt.h"
#include "api/compute/experimental/mul_reduce_scalar.h"
#include "api/compute/experimental/pack_block.h"
#include "api/compute/experimental/rmsnorm.h"
#endif
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

using std::uint32_t;

// Phase 1: this core's [M, Kc] x [Kc, N] partial. Phase 2 (cores with children): add the
// children's partials to it in the order they are consumed, keeping the running sum in DST.
// The result goes to cb_out on the root and to cb_send everywhere else.
//
// With FUSE_RMS_NORM each partial is followed by one 1x32 tile per row holding this core's
// sum(A^2) for that row in element 0, which phase 2 sums along with the partial. The root sums
// into cb_send instead, then writes rsqrt(sum / K + eps) * row to cb_out.
using namespace ckernel;
void kernel_main() {
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t Kc_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(3);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(4);
    constexpr uint32_t dst_num_tiles = get_compile_time_arg_val(5);

    const uint32_t num_children = get_arg_val<uint32_t>(0);
    const uint32_t is_root = get_arg_val<uint32_t>(1);

    constexpr uint32_t in0_cb_id = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t in1_cb_id = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t partial_cb_id = get_named_compile_time_arg_val("cb_partial");
    constexpr uint32_t recv_cb_id = get_named_compile_time_arg_val("cb_recv");
    constexpr uint32_t out_cb_id = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t send_cb_id = get_named_compile_time_arg_val("cb_send");

    constexpr uint32_t in0_num_tiles = M_tiles * Kc_tiles;
    constexpr uint32_t in1_num_tiles = Kc_tiles * N_tiles;
    constexpr uint32_t out_num_tiles = M_tiles * N_tiles;
#ifdef FUSE_RMS_NORM
    constexpr uint32_t rms_inv_k_bits = get_compile_time_arg_val(6);
    constexpr uint32_t rms_epsilon_bits = get_compile_time_arg_val(7);
    constexpr uint32_t rms_scale_cb_id = get_named_compile_time_arg_val("cb_rms_scale");
    constexpr uint32_t block_num_tiles = out_num_tiles + M_tiles;
    const uint32_t result_cb_id = send_cb_id;
#else
    constexpr uint32_t block_num_tiles = out_num_tiles;
    const uint32_t result_cb_id = is_root ? out_cb_id : send_cb_id;
#endif
    // A leaf's partial is already its result.
    const uint32_t matmul_cb_id = num_children > 0 ? partial_cb_id : result_cb_id;

    CircularBuffer in0_cb(in0_cb_id);
    CircularBuffer in1_cb(in1_cb_id);
    CircularBuffer matmul_cb(matmul_cb_id);

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb_id, in1_cb_id, matmul_cb_id);

    in0_cb.wait_front(in0_num_tiles);
    in1_cb.wait_front(in1_num_tiles);
    matmul_cb.reserve_back(block_num_tiles);
#ifdef USE_CUSTOM_MM
    constexpr bool transpose = false;
    constexpr bool split_acc = true;
    constexpr bool dense_packing = true;

    static_assert(M_tiles == 1, "custom_mm requires a single row of A");
    static_assert(Kc_tiles >= 2 && Kc_tiles <= 256 && Kc_tiles % 2 == 0);
    static_assert(N_tiles >= 1 && N_tiles <= 16);

    custom_mm_block_init_short<transpose, split_acc, dense_packing>(in0_cb_id, in1_cb_id, matmul_cb_id, N_tiles);
    pack_block_contiguous_init(matmul_cb_id);
    tile_regs_acquire();
    custom_mm_block<true, false>(in0_cb_id, in1_cb_id, 0, 0, 0, Kc_tiles, N_tiles);
    tile_regs_commit();
    tile_regs_wait();
    pack_block_contiguous(0, matmul_cb_id, N_tiles);
    tile_regs_release();
    custom_mm_block_uninit<dense_packing>();
#else
    // A is [M, Kc] row-major as 1x32 tiles, so row m's k-th tile is m * Kc_tiles + k -- the stride
    // matmul_block assumes for an rt_dim x kt_dim in0 block.
    matmul_block_init(in0_cb_id, in1_cb_id, false, out_subblock_w, out_subblock_h, Kc_tiles);
    for (uint32_t m0 = 0; m0 < M_tiles; m0 += out_subblock_h) {
        for (uint32_t n0 = 0; n0 < N_tiles; n0 += out_subblock_w) {
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kc_tiles; ++kt) {
                matmul_block(
                    in0_cb_id,
                    in1_cb_id,
                    m0 * Kc_tiles + kt,
                    kt * N_tiles + n0,
                    0,
                    false,
                    out_subblock_w,
                    out_subblock_h,
                    Kc_tiles);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t h = 0; h < out_subblock_h; ++h) {
                for (uint32_t w = 0; w < out_subblock_w; ++w) {
                    pack_tile<true>(h * out_subblock_w + w, matmul_cb_id, (m0 + h) * N_tiles + n0 + w);
                }
            }
            tile_regs_release();
        }
    }
#endif
#ifdef FUSE_RMS_NORM
    // custom_mm_block_uninit leaves the packer on a 32x32 tile stride and matmul_block_init the
    // unpacker on in1's geometry; reprogram both for the 1x32 tiles of A and of the partial.
    compute_kernel_hw_startup(in0_cb_id, in0_cb_id, matmul_cb_id);
    for (uint32_t m = 0; m < M_tiles; ++m) {
        tile_regs_acquire();
        // The multiply accumulates into DST, and acquiring it does not guarantee a cleared slot.
        fill_tile_init();
        fill_tile(0, 0.0F);
        // HiFi4 regardless of the program's fidelity: custom_mm runs LoFi, whose truncated
        // operands bias every square low.
        mul_reduce_scalar_init_fidelity<MathFidelity::HiFi4>(in0_cb_id, in0_cb_id);
        mul_reduce_scalar_tile_fidelity<MathFidelity::HiFi4, MathFidelity::HiFi4, true>(
            in0_cb_id, in0_cb_id, matmul_cb_id, Kc_tiles, 1.0F, m * Kc_tiles);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, matmul_cb_id, out_num_tiles + m);
        mul_reduce_scalar_uninit();
        tile_regs_release();
    }
#endif
    matmul_cb.push_back(block_num_tiles);
    in0_cb.pop_front(in0_num_tiles);
    in1_cb.pop_front(in1_num_tiles);

    if (num_children > 0) {
        CircularBuffer partial_cb(partial_cb_id);
        CircularBuffer recv_cb(recv_cb_id);
        CircularBuffer result_cb(result_cb_id);

        // TODO(#52395): compute_kernel_hw_startup is a call-once API; this re-init from the matmul's
        // 32x32 in1 geometry to the 1x32 partials should become a targeted reconfig.
        compute_kernel_hw_startup(partial_cb_id, recv_cb_id, result_cb_id);

        partial_cb.wait_front(block_num_tiles);
        result_cb.reserve_back(block_num_tiles);
        for (uint32_t base = 0; base < block_num_tiles; base += dst_num_tiles) {
            const uint32_t chunk = block_num_tiles - base < dst_num_tiles ? block_num_tiles - base : dst_num_tiles;
            tile_regs_acquire();
            copy_init(partial_cb_id);
            for (uint32_t t = 0; t < chunk; ++t) {
                copy_tile(partial_cb_id, base + t, t);
            }
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(recv_cb_id);
            for (uint32_t child = 0; child < num_children; ++child) {
                // Children land level by level, so the sum is consumed as the subtrees finish.
                recv_cb.wait_front((child + 1) * block_num_tiles);
                for (uint32_t t = 0; t < chunk; ++t) {
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                        recv_cb_id, child * block_num_tiles + base + t, t);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < chunk; ++t) {
                pack_tile<true>(t, result_cb_id, base + t);
            }
            tile_regs_release();
        }
        result_cb.push_back(block_num_tiles);
        partial_cb.pop_front(block_num_tiles);
        recv_cb.pop_front(num_children * block_num_tiles);
    }

#ifdef FUSE_RMS_NORM
    if (!is_root) {
        return;
    }
    CircularBuffer sum_cb(send_cb_id);
    CircularBuffer scale_cb(rms_scale_cb_id);
    CircularBuffer out_cb(out_cb_id);

    compute_kernel_hw_startup(send_cb_id, send_cb_id, rms_scale_cb_id);
    sum_cb.wait_front(block_num_tiles);

    // scale = rsqrt(sum(A^2) / K + eps), in element 0 of one tile per row.
    scale_cb.reserve_back(M_tiles);
    for (uint32_t m = 0; m < M_tiles; ++m) {
        tile_regs_acquire();
        copy_init(send_cb_id);
        copy_tile(send_cb_id, out_num_tiles + m, 0);
        add_rsqrt_tile_init();
        add_rsqrt_tile<false, VectorMode::RC_custom, 1, false, rms_inv_k_bits>(0, rms_epsilon_bits);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, rms_scale_cb_id, m);
        tile_regs_release();
    }
    scale_cb.push_back(M_tiles);
    scale_cb.wait_front(M_tiles);

    reconfig_data_format(rms_scale_cb_id, rms_scale_cb_id);
    pack_reconfig_data_format(out_cb_id);
    pack_block_contiguous_init(out_cb_id);
    for (uint32_t m = 0; m < M_tiles; ++m) {
        out_cb.reserve_back(N_tiles);
        tile_regs_acquire();
        reconfig_data_format_srca(send_cb_id, rms_scale_cb_id);
        copy_init(rms_scale_cb_id);
        copy_tile(rms_scale_cb_id, m, 0);
        reconfig_data_format_srca(rms_scale_cb_id, send_cb_id);
        rmsnorm_bcast_scalar_reuse_tiles_init_fidelity<EltwiseBinaryType::ELWMUL, N_tiles, MathFidelity::HiFi4>(
            send_cb_id);
        rmsnorm_bcast_scalar_reuse_tiles_fidelity<EltwiseBinaryType::ELWMUL, N_tiles, MathFidelity::HiFi4, true>(
            send_cb_id, m * N_tiles, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_block_contiguous(0, out_cb_id, N_tiles);
        tile_regs_release();
        out_cb.push_back(N_tiles);
    }
    scale_cb.pop_front(M_tiles);
    sum_cb.pop_front(block_num_tiles);
#endif
}
