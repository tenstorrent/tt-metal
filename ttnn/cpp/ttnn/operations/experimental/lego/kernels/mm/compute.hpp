// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Matmul compute. Port of minimal_matmul/device/kernels/compute.cpp (and, for SwiGLU, compute_metal2.cpp's
// in-K-loop path): same blocked matmul with packer L1 accumulation into the intermediate CB and the same output
// stages (copy; SwiGLU on tile-interleaved gate/up, either applied on the pack thread in the last K block when
// swiglu_in_k_loop, or as an epilogue pass). Not ported: bias, fused ternary, fused activation.
//
// Added epilogue (row_scale): output row tile i is multiplied by scale tile i (from cb_scale, filled by the in0
// block) before the output stage, on the SFPU, so the result does not depend on the kernel's math fidelity. Used to
// apply an RMSNorm's per-row 1/rms to the matmul of the un-normalized rows: (r * x) @ W = r * (x @ W).
//
// Compile-time values (A::): K_num_blocks M_block_tiles K_block_tiles N_block_tiles M_blocks_per_core
//   N_blocks_per_core subblock_h subblock_w fuse_swiglu swiglu_in_k_loop swiglu_block_float_output hw_startup
//   cb_in0 cb_in1 cb_out cb_interm row_scale cb_scale
// Runtime args: M_start_tile M_end_tile N_start_tile N_end_tile
//
// hw_startup: true for the first compute block of a kernel (compute_kernel_hw_startup must be the first compute
// call). Every block configures the unpack/math/pack state it needs at the start of run(), so blocks can follow
// each other on one TRISC.

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/operations/experimental/minimal_matmul/device/kernels/swiglu_sfpu.hpp"
#include "ttnn/operations/experimental/lego/kernels/mm/swiglu_scaled_sfpu.hpp"
#include "ttnn/operations/experimental/lego/kernels/lego.hpp"

namespace lego::mm {

template <typename A>
struct Compute {
    uint32_t M_start_tile, M_end_tile, N_start_tile, N_end_tile;

    void init() {
        M_start_tile = rt<A>(0);
        M_end_tile = rt<A>(1);
        N_start_tile = rt<A>(2);
        N_end_tile = rt<A>(3);
    }

    static void copy_and_pack_block(uint32_t in_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
        CircularBuffer cb_out(out_cb);
        reconfig_data_format_srca(in_cb);
        pack_reconfig_data_format(out_cb);
        copy_init(in_cb);
        uint32_t tile_id = 0;
        for (uint32_t m = 0; m < M_block_tiles; m++) {
            for (uint32_t n = 0; n < N_block_tiles; n++) {
                tile_regs_acquire();
                tile_regs_wait();
                copy_tile(in_cb, tile_id, 0);
                pack_tile(0, out_cb);
                tile_regs_commit();
                tile_regs_release();
                tile_id++;
            }
            cb_out.push_back(N_block_tiles);
        }
    }

    // Column tile 2p is the gate projection, 2p+1 the up projection: out tile p = silu(gate) * up.
    static void swiglu_block(uint32_t in_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
        CircularBuffer cb_out(out_cb);
        reconfig_data_format_srca(in_cb);
        pack_reconfig_data_format(out_cb);
        constexpr uint32_t GATE_DST = 0;
        constexpr uint32_t UP_DST = 1;
        const uint32_t out_N_block_tiles = N_block_tiles >> 1;
        for (uint32_t m = 0; m < M_block_tiles; m++) {
            const uint32_t row_base = m * N_block_tiles;
            for (uint32_t p = 0; p < out_N_block_tiles; p++) {
                const uint32_t gate_tile_id = row_base + (p << 1);
                tile_regs_acquire();
                copy_init(in_cb);
                copy_tile(in_cb, gate_tile_id, GATE_DST);
                copy_tile(in_cb, gate_tile_id + 1, UP_DST);
                silu_tile_init();
                silu_tile(GATE_DST);
                mul_binary_tile_init();
                mul_binary_tile(GATE_DST, UP_DST, GATE_DST);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(GATE_DST, out_cb);
                tile_regs_release();
            }
            cb_out.push_back(out_N_block_tiles);
        }
    }

    // copy_and_pack_block with each row tile multiplied by its scale tile (SFPU). Per DST acquire: the row's scale tile
    // in the last slot, up to SCALE_GROUP output tiles in the others, one init per op for the whole group.
    static constexpr uint32_t SCALE_GROUP = 6;
    static void copy_scale_pack_block(
        uint32_t in_cb, uint32_t scale_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
        CircularBuffer cb_out(out_cb);
        pack_reconfig_data_format(out_cb);
        constexpr uint32_t SCALE_DST = SCALE_GROUP;
        for (uint32_t m = 0; m < M_block_tiles; m++) {
            for (uint32_t n0 = 0; n0 < N_block_tiles; n0 += SCALE_GROUP) {
                const uint32_t g = std::min(SCALE_GROUP, N_block_tiles - n0);
                tile_regs_acquire();
                reconfig_data_format_srca(scale_cb);
                copy_init(scale_cb);
                copy_tile(scale_cb, m, SCALE_DST);
                reconfig_data_format_srca(in_cb);
                copy_init(in_cb);
                for (uint32_t i = 0; i < g; i++) {
                    copy_tile(in_cb, m * N_block_tiles + n0 + i, i);
                }
                mul_binary_tile_init();
                for (uint32_t i = 0; i < g; i++) {
                    mul_binary_tile(i, SCALE_DST, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < g; i++) {
                    pack_tile(i, out_cb);
                }
                tile_regs_release();
            }
            cb_out.push_back(N_block_tiles);
        }
    }

    // swiglu_block with gate and up both multiplied by the row's scale first: silu(r*g) * (r*u). Per DST acquire:
    // the scale tile and up to SCALE_GROUP / 2 gate/up pairs.
    static void swiglu_scale_block(
        uint32_t in_cb, uint32_t scale_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
        CircularBuffer cb_out(out_cb);
        pack_reconfig_data_format(out_cb);
        constexpr uint32_t PAIRS = SCALE_GROUP / 2;
        constexpr uint32_t SCALE_DST = SCALE_GROUP;
        const uint32_t out_N_block_tiles = N_block_tiles >> 1;
        for (uint32_t m = 0; m < M_block_tiles; m++) {
            const uint32_t row_base = m * N_block_tiles;
            for (uint32_t p0 = 0; p0 < out_N_block_tiles; p0 += PAIRS) {
                const uint32_t g = std::min(PAIRS, out_N_block_tiles - p0);
                tile_regs_acquire();
                reconfig_data_format_srca(scale_cb);
                copy_init(scale_cb);
                copy_tile(scale_cb, m, SCALE_DST);
                reconfig_data_format_srca(in_cb);
                copy_init(in_cb);
                for (uint32_t i = 0; i < 2 * g; i++) {
                    copy_tile(in_cb, row_base + 2 * p0 + i, i);  // gate 2i, up 2i+1
                }
                mul_binary_tile_init();
                for (uint32_t i = 0; i < 2 * g; i++) {
                    mul_binary_tile(i, SCALE_DST, i);
                }
                silu_tile_init();
                for (uint32_t i = 0; i < g; i++) {
                    silu_tile(2 * i);
                }
                mul_binary_tile_init();
                for (uint32_t i = 0; i < g; i++) {
                    mul_binary_tile(2 * i, 2 * i + 1, 2 * i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < g; i++) {
                    pack_tile(2 * i, out_cb);
                }
                tile_regs_release();
            }
            cb_out.push_back(out_N_block_tiles);
        }
    }

    static void matmul_blocks(
        uint32_t in0_cb,
        uint32_t in1_cb,
        uint32_t out_cb,
        uint32_t M_block_tiles,
        uint32_t N_block_tiles,
        uint32_t full_N_block_tiles,
        uint32_t K_block_tiles,
        uint32_t subblock_h,
        uint32_t subblock_w) {
        uint32_t in0_index_offset = 0;
        for (uint32_t M_start = 0; M_start < M_block_tiles; M_start += subblock_h) {
            uint32_t in1_index_offset = 0;
            for (uint32_t N_start = 0; N_start < N_block_tiles; N_start += subblock_w) {
                tile_regs_acquire();
                uint32_t dst_index = 0;
                uint32_t in0_index = in0_index_offset;
                uint32_t in1_index = in1_index_offset;
                for (uint32_t inner_dim = 0; inner_dim < K_block_tiles; inner_dim++) {
                    matmul_block(
                        in0_cb, in1_cb, in0_index, in1_index, dst_index, false, subblock_w, subblock_h, K_block_tiles);
                    in0_index++;
                    in1_index += full_N_block_tiles;
                }
                tile_regs_commit();
                tile_regs_wait();
                uint32_t write_dst_index = 0;
                for (uint32_t h = 0; h < subblock_h; h++) {
                    uint32_t h_tile_id = M_start + h;
                    for (uint32_t w = 0; w < subblock_w; w++) {
                        uint32_t out_tile_id = h_tile_id * full_N_block_tiles + N_start + w;
                        pack_tile<true>(write_dst_index, out_cb, out_tile_id);
                        write_dst_index++;
                        dst_index++;
                    }
                }
                tile_regs_release();
                in1_index_offset += subblock_w;
            }
            in0_index_offset += subblock_h * K_block_tiles;
        }
    }

    // compute_metal2.cpp matmul_blocks_swiglu: the last K block of a SwiGLU output block. Accumulate the last K
    // block in DST, add the packer-accumulated partials of blocks 0..K-2 with one dest-reuse add, then run
    // silu(gate) * up on the pack thread and pack straight into out (half-width block).
    // scaled: also load each subblock row's scale tile (cb_scale tile M_start + h) into DST slot
    // subblock_h * subblock_w + h, and run silu(s * gate) * (s * up) on the pack thread.
    template <bool block_float_output, bool scaled>
    static void matmul_blocks_swiglu(
        uint32_t in0_cb,
        uint32_t in1_cb,
        uint32_t interm_cb,
        uint32_t out_cb,
        uint32_t M_block_tiles,
        uint32_t N_block_tiles,
        uint32_t full_N_block_tiles,
        uint32_t K_block_tiles,
        uint32_t subblock_h,
        uint32_t subblock_w,
        bool add_partials,
        uint32_t scale_cb = 0) {
        const uint32_t out_full_N_block_tiles = full_N_block_tiles >> 1;
        uint32_t in0_index_offset = 0;
        for (uint32_t M_start = 0; M_start < M_block_tiles; M_start += subblock_h) {
            uint32_t in1_index_offset = 0;
            for (uint32_t N_start = 0; N_start < N_block_tiles; N_start += subblock_w) {
                tile_regs_acquire();
                uint32_t in0_index = in0_index_offset;
                uint32_t in1_index = in1_index_offset;
                for (uint32_t inner_dim = 0; inner_dim < K_block_tiles; inner_dim++) {
                    matmul_block(in0_cb, in1_cb, in0_index, in1_index, 0, false, subblock_w, subblock_h, K_block_tiles);
                    in0_index++;
                    in1_index += full_N_block_tiles;
                }
                if (add_partials) {
                    reconfig_data_format_srca(in1_cb, interm_cb);
                    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(interm_cb);
                    uint32_t dst = 0;
                    for (uint32_t h = 0; h < subblock_h; h++) {
                        for (uint32_t w = 0; w < subblock_w; w++) {
                            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                                interm_cb, (M_start + h) * full_N_block_tiles + N_start + w, dst++);
                        }
                    }
                    reconfig_data_format_srca(interm_cb, in1_cb);
                    matmul_block_init(in0_cb, in1_cb, false, subblock_w, subblock_h, K_block_tiles);
                }
                if constexpr (scaled) {
                    reconfig_data_format_srca(in1_cb, scale_cb);
                    copy_init(scale_cb);
                    for (uint32_t h = 0; h < subblock_h; h++) {
                        copy_tile(scale_cb, M_start + h, subblock_h * subblock_w + h);
                    }
                    reconfig_data_format_srca(scale_cb, in1_cb);
                    matmul_block_init(in0_cb, in1_cb, false, subblock_w, subblock_h, K_block_tiles);
                }
                tile_regs_commit();
                PACK(TTI_SEMWAIT(
                    p_stall::STALL_TDMA | p_stall::STALL_CFG,
                    semaphore::t6_sem(semaphore::MATH_PACK),
                    p_stall::STALL_ON_ZERO));
                PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
                for (uint32_t h = 0; h < subblock_h; h++) {
                    for (uint32_t w = 0; w < subblock_w; w += 2) {
                        const uint32_t gate = h * subblock_w + w;
                        if constexpr (scaled) {
                            PACK((lego_swiglu_scaled_tile<block_float_output>(
                                gate, gate + 1, subblock_h * subblock_w + h)));
                        } else {
                            PACK((minimal_matmul_swiglu_tile<block_float_output>(gate, gate + 1, gate)));
                        }
                    }
                }
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                for (uint32_t h = 0; h < subblock_h; h++) {
                    for (uint32_t w = 0; w < subblock_w; w += 2) {
                        pack_tile<true>(
                            h * subblock_w + w, out_cb, (M_start + h) * out_full_N_block_tiles + ((N_start + w) >> 1));
                    }
                }
                tile_regs_release();
                in1_index_offset += subblock_w;
            }
            in0_index_offset += subblock_h * K_block_tiles;
        }
    }

    void run() {
        constexpr uint32_t in0_cb = A::cb_in0, in1_cb = A::cb_in1, out_cb = A::cb_out, intermediate_cb = A::cb_interm;
        CircularBuffer cb_in0(in0_cb);
        CircularBuffer cb_in1(in1_cb);
        CircularBuffer cb_out(out_cb);
        CircularBuffer cb_intermediate(intermediate_cb);

        if constexpr (A::hw_startup) {
            compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb, in1_cb, intermediate_cb);
        }
        matmul_init(in0_cb, in1_cb);
        if constexpr (A::swiglu_in_k_loop) {
            PACK((minimal_matmul_swiglu_tile_init()));
        }

        constexpr uint32_t in0_block_num_tiles = A::M_block_tiles * A::K_block_tiles;
        constexpr uint32_t in1_block_num_tiles = A::K_block_tiles * A::N_block_tiles;
        constexpr uint32_t out_block_num_tiles = A::M_block_tiles * A::N_block_tiles;

        bool reuse_in0_block = false;
        for (uint32_t m_block_iter = 0; m_block_iter < A::M_blocks_per_core; m_block_iter++) {
            uint32_t m_tile, m_tile_end;
            if constexpr (A::m_batches > 1) {  // block b: this core's rows of batch b
                m_tile = M_start_tile + m_block_iter * A::m_batch_tiles;
                m_tile_end = M_end_tile + m_block_iter * A::m_batch_tiles;
            } else {
                m_tile = M_start_tile + m_block_iter * A::M_block_tiles;
                m_tile_end = std::min(m_tile + A::M_block_tiles, M_end_tile);
            }
            uint32_t current_M_block_tiles = m_tile_end - m_tile;
            uint32_t current_subblock_h = std::min(current_M_block_tiles, A::subblock_h);

            for (uint32_t n_block_iter = 0; n_block_iter < A::N_blocks_per_core; n_block_iter++) {
                uint32_t n_tile = N_start_tile + n_block_iter * A::N_block_tiles;
                uint32_t n_tile_end = std::min(n_tile + A::N_block_tiles, N_end_tile);
                uint32_t current_N_block_tiles = n_tile_end - n_tile;
                uint32_t current_subblock_w = std::min(current_N_block_tiles, A::subblock_w);

                reconfig_data_format(in1_cb, in0_cb);
                pack_reconfig_data_format(intermediate_cb);
                matmul_block_init(in0_cb, in1_cb, false, current_subblock_w, current_subblock_h, A::K_block_tiles);
                cb_intermediate.reserve_back(out_block_num_tiles);
                for (uint32_t k_block = 0; k_block < A::K_num_blocks; k_block++) {
                    cb_in0.wait_front(in0_block_num_tiles);
                    cb_in1.wait_front(in1_block_num_tiles);
                    if (A::swiglu_in_k_loop && k_block == A::K_num_blocks - 1) {
                        if (A::K_num_blocks > 1) {  // publish the partials of K blocks 0..K-2 for the add
                            cb_intermediate.push_back(out_block_num_tiles);
                            cb_intermediate.wait_front(out_block_num_tiles);
                        }
                        PACK((llk_pack_reconfig_l1_acc(0)));
                        pack_reconfig_data_format(out_cb);
                        cb_out.reserve_back(out_block_num_tiles >> 1);
                        if constexpr (A::row_scale) {
                            if (n_block_iter == 0) {
                                CircularBuffer(A::cb_scale).wait_front(A::M_block_tiles);
                            }
                        }
                        matmul_blocks_swiglu<A::swiglu_block_float_output, A::row_scale>(
                            in0_cb,
                            in1_cb,
                            intermediate_cb,
                            out_cb,
                            current_M_block_tiles,
                            current_N_block_tiles,
                            A::N_block_tiles,
                            A::K_block_tiles,
                            current_subblock_h,
                            current_subblock_w,
                            A::K_num_blocks > 1,
                            A::cb_scale);
                        if constexpr (A::row_scale) {
                            if (n_block_iter == A::N_blocks_per_core - 1) {
                                CircularBuffer(A::cb_scale).pop_front(A::M_block_tiles);
                            }
                        }
                        cb_out.push_back(out_block_num_tiles >> 1);
                        if (A::K_num_blocks > 1) {
                            cb_intermediate.pop_front(out_block_num_tiles);
                        }
                    } else {
                        matmul_blocks(
                            in0_cb,
                            in1_cb,
                            intermediate_cb,
                            current_M_block_tiles,
                            current_N_block_tiles,
                            A::N_block_tiles,
                            A::K_block_tiles,
                            current_subblock_h,
                            current_subblock_w);
                    }
                    if (k_block == A::K_num_blocks - 1 && n_block_iter < A::N_blocks_per_core - 1) {
                        reuse_in0_block = true;  // striding N next: keep in0
                    }
                    if (!reuse_in0_block) {
                        cb_in0.pop_front(in0_block_num_tiles);
                    }
                    cb_in1.pop_front(in1_block_num_tiles);
                    reuse_in0_block = false;
                    if (k_block == 0) {
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }
                }
                if constexpr (!A::swiglu_in_k_loop) {
                    cb_intermediate.push_back(out_block_num_tiles);
                }
                PACK((llk_pack_reconfig_l1_acc(0)));

                if constexpr (A::row_scale && !A::swiglu_in_k_loop) {
                    CircularBuffer cb_scale(A::cb_scale);
                    if (n_block_iter == 0) {
                        cb_scale.wait_front(A::M_block_tiles);
                    }
                    cb_intermediate.wait_front(out_block_num_tiles);
                    if constexpr (A::fuse_swiglu) {
                        cb_out.reserve_back(out_block_num_tiles >> 1);
                        swiglu_scale_block(intermediate_cb, A::cb_scale, out_cb, A::M_block_tiles, A::N_block_tiles);
                    } else {
                        cb_out.reserve_back(out_block_num_tiles);
                        copy_scale_pack_block(intermediate_cb, A::cb_scale, out_cb, A::M_block_tiles, A::N_block_tiles);
                    }
                    cb_intermediate.pop_front(out_block_num_tiles);
                    if (n_block_iter == A::N_blocks_per_core - 1) {
                        cb_scale.pop_front(A::M_block_tiles);
                    }
                } else if constexpr (A::swiglu_in_k_loop) {
                    // SwiGLU was applied in the last K block.
                } else if constexpr (A::fuse_swiglu) {
                    cb_out.reserve_back(out_block_num_tiles >> 1);
                    cb_intermediate.wait_front(out_block_num_tiles);
                    swiglu_block(intermediate_cb, out_cb, A::M_block_tiles, A::N_block_tiles);
                    cb_intermediate.pop_front(out_block_num_tiles);
                } else {
                    cb_out.reserve_back(out_block_num_tiles);
                    cb_intermediate.wait_front(out_block_num_tiles);
                    copy_and_pack_block(intermediate_cb, out_cb, A::M_block_tiles, A::N_block_tiles);
                    cb_intermediate.pop_front(out_block_num_tiles);
                }
            }
        }
    }

    void teardown() {}
};

}  // namespace lego::mm
