// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// moe_down_proj accumulate_wave, raw LLK (bypasses compute_kernel_lib::matmul_block and the ckernel::matmul_block
// API). Same tiles, same per-tile k order and the same packer L1 accumulation per round as the helper path, so
// the output is bit-identical. In the compute-alone regime the unpack and math issue chain binds, not the FPU or
// the unpacker; per DEST acquire this path removes, against matmul_block at 1 x core_n_tiles:
//   unpack: the per-k-step config context (wait, RISC-V base-address writes, STALLWAIT, semaphores). One context per
//     tile row; the k_block steps replay [UNPACR srcB, srcB base += x tile, core_n_tiles x (UNPACR srcA, srcA base
//     += w tile)] with CFGSHIFTMASK adds (Tensix side, no RDCFG / WRCFG / STALLWAIT).
//   math: the per-tile DEST offset write (SETC16). Tiles 0 .. ct-2 of a k step end on ADDR_MOD_3 (dest carriage
//     return += one tile) instead of ADDR_MOD_5. The firmware throttle flag is sampled once per row, not per k step.
//   sync: DEST_AUTO_LIMIT / core_n_tiles tile rows per acquire (one pack handoff per 2 rows at core_n_tiles 2).

#pragma once

namespace moe_accumulate {

// The unpack replay holds 3 + 3 * ct instructions per config context in half the replay buffer.
template <uint32_t core_n_tiles>
constexpr bool fits = core_n_tiles <= 4 && core_n_tiles <= compute_kernel_lib::DEST_AUTO_LIMIT;

#ifdef TRISC_MATH
// Re-programs the math MOP when the firmware throttle flag flips (as matmul_block_math_dynamic_throttle does).
template <uint32_t ct>
ALWI bool throttle_sample(uint32_t cb_x, uint32_t cb_w) {
    const bool throttle = (*throttle_ptr % 2) != 0;
    if (throttle && throttled_mop_status != 1) {
        llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE_MAX>(cb_x, cb_w, true, ct, 1);
        throttled_mop_status = 1;
    } else if (!throttle && throttled_mop_status != 0) {
        llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(cb_x, cb_w, true, ct, 1);
        throttled_mop_status = 0;
    }
    return throttle;
}

// One tile row: k_block steps into DEST tiles [dt, dt + ct).
template <uint32_t kt, uint32_t ct>
ALWI void math_row(uint32_t cb_x, uint32_t cb_w, uint32_t dt) {
    if (throttle_sample<ct>(cb_x, cb_w)) {
        for (uint32_t kk = 0; kk < kt; ++kk) {
            _llk_math_matmul_<MATH_FIDELITY, MM_THROTTLE_MAX>(dt, ct, 1);
        }
        return;
    }
    if constexpr (ckernel::math::is_high_fidelity(MATH_FIDELITY) || MM_THROTTLE != 0) {
        // The split below assumes the single-phase LoFi MOP.
        for (uint32_t kk = 0; kk < kt; ++kk) {
            _llk_math_matmul_<MATH_FIDELITY, MM_THROTTLE>(dt, ct, 1);
        }
        return;
    }
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::get_dest_buffer_base() + (dt << 6));
#pragma GCC unroll 16
    for (uint32_t kk = 0; kk < kt; ++kk) {
#pragma GCC unroll 4
        for (uint32_t c = 0; c + 1 < ct; ++c) {
            TTI_REPLAY(ckernel::math::replay_buf_offset, 15, 0, 0);
            TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_3, 0);
        }
        TTI_MOP(1, 0, 0);
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    }
}

ALWI void math_init() {
    ckernel::addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 1},
        .dest = {.incr = 64, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_3);
}
#endif

#ifdef TRISC_UNPACK
// Replay: config context c at [c * len, (c + 1) * len).
template <uint32_t ct>
ALWI void unpack_init(uint32_t cb_x, uint32_t cb_w) {
    constexpr uint32_t len = 3 + 3 * ct;
    load_replay_buf(0, 2 * len, [] {
        TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 1, THCON_SEC1_REG3_Base_address_ADDR32);
        TTI_NOP;
        for (uint32_t c = 0; c < ct; ++c) {
            TTI_UNPACR(SrcA, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0, THCON_SEC0_REG3_Base_address_ADDR32);
            TTI_NOP;
        }
        TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 1, THCON_SEC1_REG3_Base_cntx1_address_ADDR32);
        TTI_NOP;
        for (uint32_t c = 0; c < ct; ++c) {
            TTI_UNPACR(SrcA, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
            TTI_NOP;
        }
    });
    volatile uint32_t* cfg = get_cfg_pointer();
    cfg[SCRATCH_SEC0_val_ADDR32] = get_local_cb_interface(cb_w).fifo_page_size;
    cfg[SCRATCH_SEC1_val_ADDR32] = get_local_cb_interface(cb_x).fifo_page_size;
}

// One tile row: k_block steps of [x tile, ct w tiles] from x_addr / w_addr (unpacker 16 B units).
template <uint32_t kt, uint32_t ct>
ALWI void unpack_row(uint32_t x_addr, uint32_t w_addr) {
    constexpr uint32_t len = 3 + 3 * ct;
    volatile uint32_t* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    // SEC0 (srcA) = in1 = w, SEC1 (srcB) = in0 = x.
    _llk_unpack_configure_addresses_(w_addr, x_addr, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    const uint32_t start = unp_cfg_context == 0 ? 0 : len;
#pragma GCC unroll 16
    for (uint32_t kk = 0; kk < kt; ++kk) {
        TT_REPLAY(start, len, 0, 0);
    }
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
#endif

template <
    uint32_t k_block_tiles,
    uint32_t num_k_rounds,
    uint32_t core_n_tiles,
    uint32_t core_row_tiles_max,
    typename Buf>
ALWI void accumulate_wave(
    Buf& x_buf,
    Buf& w_buf,
    const MoePlan& plan,
    uint32_t row_lo,
    uint32_t row_hi,
    uint32_t my_rows,
    uint32_t last_slot,
    uint32_t cb_x,
    uint32_t cb_w,
    uint32_t cb_partials) {
    static_assert(fits<core_n_tiles>);
    constexpr uint32_t w_chunk_tiles = k_block_tiles * core_n_tiles;
    constexpr uint32_t x_round_tiles = core_row_tiles_max * k_block_tiles;
    constexpr uint32_t acq_rows = compute_kernel_lib::DEST_AUTO_LIMIT / core_n_tiles;

    // The first matmul of a wave follows the untilize (or boot).
    ckernel::reconfig_data_format(cb_w, cb_x);
    ckernel::pack_reconfig_data_format(cb_partials);
    ckernel::matmul_block_init(cb_x, cb_w, true, core_n_tiles, 1, k_block_tiles);
    MATH((math_init()));
    UNPACK((unpack_init<core_n_tiles>(cb_x, cb_w)));

    MaybeDeviceZoneScope("compute_accumulate");
    for (uint32_t round = 0; round < num_k_rounds; ++round) {
        if (my_rows > 0) {
            MaybeDeviceZoneScope("compute_wait_inputs");
            x_buf.wait_front(my_rows * k_block_tiles);
            w_buf.wait_front(last_slot * w_chunk_tiles);
        }
        pack_reconfig_l1_acc(round > 0 ? 1 : 0);

        uint32_t x_addr = 0, x_step = 0, w_base = 0, w_step = 0;
        UNPACK(({
            x_addr = get_local_cb_interface(cb_x).fifo_rd_ptr - 1;
            x_step = get_local_cb_interface(cb_x).fifo_page_size;
            w_base = get_local_cb_interface(cb_w).fifo_rd_ptr - 1;
            w_step = get_local_cb_interface(cb_w).fifo_page_size;
        }));

        for (uint32_t e = 0; e < MOE_NUM_EXPERTS; ++e) {
            uint32_t first;
            const uint32_t rows = plan.expert_rows_in(e, row_lo, row_hi, &first);
            if (rows == 0) {
                continue;
            }
            const uint32_t w_addr = w_base + e * w_chunk_tiles * w_step;
            uint32_t out_tile = (first - row_lo) * core_n_tiles;
            for (uint32_t r = 0; r < rows; r += acq_rows) {
                const uint32_t n_rows = rows - r < acq_rows ? rows - r : acq_rows;
                tile_regs_acquire();
                for (uint32_t rr = 0; rr < n_rows; ++rr) {
                    MATH((math_row<k_block_tiles, core_n_tiles>(cb_x, cb_w, rr * core_n_tiles)));
                    UNPACK((unpack_row<k_block_tiles, core_n_tiles>(x_addr, w_addr)));
                    UNPACK((x_addr += k_block_tiles * x_step));
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < n_rows * core_n_tiles; ++t) {
                    pack_tile<true>(t, cb_partials, out_tile + t);
                }
                tile_regs_release();
                out_tile += n_rows * core_n_tiles;
            }
        }
        // release_round: the whole nominal round.
        x_buf.wait_front(x_round_tiles);
        x_buf.pop_front(x_round_tiles);
        w_buf.wait_front(MOE_NUM_EXPERTS * w_chunk_tiles);
        w_buf.pop_front(MOE_NUM_EXPERTS * w_chunk_tiles);
    }
}

}  // namespace moe_accumulate
