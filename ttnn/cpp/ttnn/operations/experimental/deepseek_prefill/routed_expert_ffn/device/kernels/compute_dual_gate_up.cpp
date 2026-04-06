// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Dual gate+up matmul compute kernel with fused SiLU activation.
// Computes act_out = silu(x @ gate_proj) * (x @ up_proj) in L1,
// avoiding the DRAM round-trip for intermediate gate_out / up_out.

#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"

// ── matmul_blocks_with_offset ───────────────────────────────────────────────
void matmul_blocks_with_offset(
    const uint32_t in0_cb,
    const uint32_t in1_cb,
    const uint32_t out_cb,
    const uint32_t Mt_block_size,
    const uint32_t Nt_block_size,
    const uint32_t full_N_local,
    const uint32_t n_tile_offset,
    const uint32_t Kt_block_size,
    const uint32_t subblock_h,
    const uint32_t subblock_w) {
    uint32_t in0_index_offset = 0;

    for (uint32_t M_start = 0; M_start < Mt_block_size; M_start += subblock_h) {
        uint32_t in1_index_offset = 0;
        for (uint32_t N_start = 0; N_start < Nt_block_size; N_start += subblock_w) {
            tile_regs_acquire();

            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < Kt_block_size; inner_dim++) {
                matmul_block(
                    in0_cb, in1_cb, in0_index, in1_index, dst_index, false, subblock_w, subblock_h, Kt_block_size);
                in0_index++;
                in1_index += Nt_block_size;
            }

            tile_regs_commit();
            tile_regs_wait();

            uint32_t write_dst_index = 0;
            for (uint32_t h = 0; h < subblock_h; h++) {
                uint32_t h_tile = M_start + h;
                for (uint32_t w = 0; w < subblock_w; w++) {
                    uint32_t w_tile = n_tile_offset + N_start + w;
                    uint32_t tile_id = h_tile * full_N_local + w_tile;
                    pack_tile<true>(write_dst_index, out_cb, tile_id);
                    write_dst_index++;
                }
            }

            tile_regs_release();
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * Kt_block_size;
    }
}

// ── fuse_silu_mul ────────────────────────────────────────────────────────────
// Pass 1: silu(gate_interm) → silu_gate_cb
// Pass 2: silu_gate_cb * up_interm → act_out_cb
// Uses explicit tile_regs_commit/wait between MATH (SFPU/FPU) and PACK.
void fuse_silu_mul(
    const uint32_t gate_interm,
    const uint32_t up_interm,
    const uint32_t silu_gate_cb,
    const uint32_t act_out_cb,
    const uint32_t num_tiles) {
    // ── Pass 1: silu(gate) ─────────────────────────────────────────────────
    copy_tile_to_dst_init_short(gate_interm);
    reconfig_data_format_srca(gate_interm);
    pack_reconfig_data_format(silu_gate_cb);
    silu_tile_init();

    cb_wait_front(gate_interm, num_tiles);
    cb_reserve_back(silu_gate_cb, num_tiles);

    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        copy_tile(gate_interm, i, 0);
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, silu_gate_cb);
        tile_regs_release();
    }

    cb_push_back(silu_gate_cb, num_tiles);
    cb_pop_front(gate_interm, num_tiles);

    // ── Pass 2: silu_gate * up ─────────────────────────────────────────────
    mul_tiles_init(silu_gate_cb, up_interm);
    reconfig_data_format(silu_gate_cb, up_interm);
    pack_reconfig_data_format(act_out_cb);

    cb_wait_front(silu_gate_cb, num_tiles);
    cb_wait_front(up_interm, num_tiles);
    cb_reserve_back(act_out_cb, num_tiles);

    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        mul_tiles(silu_gate_cb, up_interm, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, act_out_cb);
        tile_regs_release();
    }

    cb_push_back(act_out_cb, num_tiles);
    cb_pop_front(silu_gate_cb, num_tiles);
    cb_pop_front(up_interm, num_tiles);
}

// ── kernel_main ─────────────────────────────────────────────────────────────
void kernel_main() {
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t m_blocks_local = get_compile_time_arg_val(1);
    constexpr uint32_t n_blocks_local = get_compile_time_arg_val(2);
    constexpr uint32_t Mt_block_size = get_compile_time_arg_val(3);
    constexpr uint32_t Kt_block_size = get_compile_time_arg_val(4);
    constexpr uint32_t Nt_block_size = get_compile_time_arg_val(5);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(6);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(7);

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_gate_cb = tt::CBIndex::c_1;
    constexpr uint32_t in1_up_cb = tt::CBIndex::c_2;
    constexpr uint32_t gate_interm = tt::CBIndex::c_3;
    constexpr uint32_t up_interm = tt::CBIndex::c_4;
    constexpr uint32_t silu_gate_cb = tt::CBIndex::c_5;
    constexpr uint32_t act_out_cb = tt::CBIndex::c_6;

    constexpr uint32_t in0_block_size = Mt_block_size * Kt_block_size;
    constexpr uint32_t in1_block_size = Kt_block_size * Nt_block_size;
    constexpr uint32_t full_N_local = n_blocks_local * Nt_block_size;
    constexpr uint32_t full_out_tiles = Mt_block_size * full_N_local;

    mm_init(in0_cb, in1_gate_cb, gate_interm);

    for (uint32_t m = 0; m < m_blocks_local; m++) {
        cb_reserve_back(gate_interm, full_out_tiles);
        cb_reserve_back(up_interm, full_out_tiles);

        for (uint32_t k = 0; k < K_num_blocks; k++) {
            cb_wait_front(in0_cb, in0_block_size);

            for (uint32_t n = 0; n < n_blocks_local; n++) {
                uint32_t n_tile_offset = n * Nt_block_size;

                mm_block_init_short(in0_cb, in1_gate_cb, false, subblock_w, subblock_h, Kt_block_size);
                reconfig_data_format(in1_gate_cb, in0_cb);
                pack_reconfig_data_format(gate_interm);

                cb_wait_front(in1_gate_cb, in1_block_size);
                matmul_blocks_with_offset(
                    in0_cb,
                    in1_gate_cb,
                    gate_interm,
                    Mt_block_size,
                    Nt_block_size,
                    full_N_local,
                    n_tile_offset,
                    Kt_block_size,
                    subblock_h,
                    subblock_w);
                cb_pop_front(in1_gate_cb, in1_block_size);

                mm_block_init_short(in0_cb, in1_up_cb, false, subblock_w, subblock_h, Kt_block_size);
                reconfig_data_format(in1_up_cb, in0_cb);
                pack_reconfig_data_format(up_interm);

                cb_wait_front(in1_up_cb, in1_block_size);
                matmul_blocks_with_offset(
                    in0_cb,
                    in1_up_cb,
                    up_interm,
                    Mt_block_size,
                    Nt_block_size,
                    full_N_local,
                    n_tile_offset,
                    Kt_block_size,
                    subblock_h,
                    subblock_w);
                cb_pop_front(in1_up_cb, in1_block_size);
            }

            cb_pop_front(in0_cb, in0_block_size);

            if (k == 0) {
                PACK((llk_pack_reconfig_l1_acc(1)));
            }
        }

        PACK((llk_pack_reconfig_l1_acc(0)));

        cb_push_back(gate_interm, full_out_tiles);
        cb_push_back(up_interm, full_out_tiles);

        fuse_silu_mul(gate_interm, up_interm, silu_gate_cb, act_out_cb, full_out_tiles);
    }
}
