// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Perf experiment (compute_floor): lean raw-LLK accumulate_wave for moe_down_proj.
//
// Bypasses compute_kernel_lib::matmul_block AND the ckernel::matmul_block API wrapper. Same LLK instruction
// stream per k step (one _llk_unpack_AB_matmul_ + one _llk_math_matmul_ at 1 x core_n_tiles), same DEST
// acquire granularity (one tile row x core_n_tiles, k_block steps), same TileRowMajor pack with packer L1
// accumulation, so the fp32 summation order is unchanged (bit-exact).
//
// Why raw: in the compute-alone regime both the unpack and the math TRISC RISC-V are on the critical path
// (8 nops per k step on either thread adds 16-29 us). Per k step the API wrapper re-derives, on the unpack
// RISC-V: operand ids, partial-face flags, CB rd_ptr / page size loads and tile-index multiplies; on the math
// RISC-V: a volatile L1 read of the firmware throttle flag. This path keeps running L1 addresses in registers
// (one add per k step) and samples the throttle flag once per DEST acquire.
//
// Knobs:
//   CF_UNROLL       math: per tile one SETC16 (precomputed DEST offset) + MOP, fully unrolled k loop; unpack:
//                   unrolled k loop.
//   CF_THROTTLE_PER_CALL  keep the per-k-step throttle read (attribution).
//   CF_DIAG_NO_SETC16 diagnostic (with CF_UNROLL): one DEST offset write per sub-block, all tiles land on tile 0.
//   CF_DIAG_NO_UNPACR diagnostic (with CF_UNPACK_REPLAY): UNPACR -> UNPACR_NOP zero-src + dvalid (math rate).
//   CF_ROWS=N       tile rows per DEST acquire, each row its own k loop (default: DEST half / core_n_tiles).
//   CF_DIAG_NO_PACK   diagnostic: no pack_tile (DEST sync kept).
//   CF_DEST_RWC     (with CF_UNROLL) advance DEST between the ct tiles of a k step with the dest RWC carriage
//                   return (ADDR_MOD_3) instead of a SETC16 DEST offset write per tile.
//   CF_UNPACK_REPLAY  unpack: one config context per DEST acquire instead of one per k step. The k_block steps
//                   replay [UNPACR srcB, base += x tile, ct x (UNPACR srcA, base += w tile)] from the replay
//                   buffer; the base increments are CFGSHIFTMASK adds of SCRATCH_SEC0/1 (Tensix side, no
//                   RDCFG / WRCFG / STALLWAIT and no RISC-V cfg write per k step). Same tiles, same order.

#pragma once

#ifdef TRISC_MATH
template <bool throttled, uint32_t ct>
ALWI void cf_math_kstep(uint32_t dt) {
    if constexpr (throttled) {
        _llk_math_matmul_<MATH_FIDELITY, MM_THROTTLE_MAX>(dt, ct, 1);
    } else {
        _llk_math_matmul_<MATH_FIDELITY, MM_THROTTLE>(dt, ct, 1);
    }
}

// Re-programs the math MOP when the firmware throttle flag flips (same as matmul_block_math_dynamic_throttle).
template <uint32_t ct>
ALWI bool cf_math_throttle_sample(uint32_t cb_x, uint32_t cb_w) {
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

template <uint32_t kt, uint32_t ct>
ALWI void cf_math_subblock(uint32_t cb_x, uint32_t cb_w, uint32_t dt) {
#if defined(CF_THROTTLE_PER_CALL)
    for (uint32_t kk = 0; kk < kt; ++kk) {
        if (cf_math_throttle_sample<ct>(cb_x, cb_w)) {
            cf_math_kstep<true, ct>(dt);
        } else {
            cf_math_kstep<false, ct>(dt);
        }
    }
#else
    if (cf_math_throttle_sample<ct>(cb_x, cb_w)) {
        for (uint32_t kk = 0; kk < kt; ++kk) {
            cf_math_kstep<true, ct>(dt);
        }
        return;
    }
#if defined(CF_UNROLL) && defined(CF_DEST_RWC) && MM_THROTTLE == 0
    if constexpr (ckernel::math::is_high_fidelity(MATH_FIDELITY)) {
        // The split below assumes the single-phase LoFi MOP; fidelity phases keep the per-tile LLK path.
        for (uint32_t kk = 0; kk < kt; ++kk) {
            cf_math_kstep<false, ct>(dt);
        }
        return;
    }
    // One DEST offset write per row sub-block. Tiles 0 .. ct-2 of a k step end with ADDR_MOD_3 (dest CR += 64:
    // the next tile's base, srcA/srcB counters cleared) instead of ADDR_MOD_5; the last tile runs the stock MOP
    // (ADDR_MOD_5 returns dest to the sub-block base).
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::get_dest_buffer_base() + (dt << 6));
#pragma GCC unroll 16
    for (uint32_t kk = 0; kk < kt; ++kk) {
#pragma GCC unroll 8
        for (uint32_t c = 0; c + 1 < ct; ++c) {
            TTI_REPLAY(ckernel::math::replay_buf_offset, 15, 0, 0);
            TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_3, 0);
        }
        TTI_MOP(1, 0, 0);
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    }
#elif defined(CF_UNROLL) && MM_THROTTLE == 0
    const uint32_t d0 = ckernel::get_dest_buffer_base() + (dt << 6);
#pragma GCC unroll 16
    for (uint32_t kk = 0; kk < kt; ++kk) {
#pragma GCC unroll 8
        for (uint32_t c = 0; c < ct; ++c) {
#ifdef CF_DIAG_NO_SETC16
            if (kk == 0 && c == 0) {
                TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, d0);
            }
#else
            TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, d0 + (c << 6));
#endif
            TTI_MOP(1, 0, 0);
        }
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    }
#else
    for (uint32_t kk = 0; kk < kt; ++kk) {
        cf_math_kstep<false, ct>(dt);
    }
#endif
#endif
}
#endif

#ifdef TRISC_UNPACK
// Replay layout: context c at [c * len, (c + 1) * len), len = 3 + 3 * ct.
template <uint32_t ct>
ALWI void cf_unpack_replay_load() {
    static_assert(3 + 3 * ct <= 16, "replay sequence must fit half the unpack replay buffer");
    constexpr uint32_t len = 3 + 3 * ct;
#ifdef CF_DIAG_NO_UNPACR
#define CF_UNPACR_A \
    TTI_UNPACR_NOP( \
        SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC)
#define CF_UNPACR_B \
    TTI_UNPACR_NOP( \
        SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC)
#else
#define CF_UNPACR_A TTI_UNPACR(SrcA, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1)
#define CF_UNPACR_B TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1)
#endif
    load_replay_buf(0, 2 * len, [] {
        CF_UNPACR_B;
        TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 1, THCON_SEC1_REG3_Base_address_ADDR32);
        TTI_NOP;
        for (uint32_t c = 0; c < ct; ++c) {
            CF_UNPACR_A;
            TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0, THCON_SEC0_REG3_Base_address_ADDR32);
            TTI_NOP;
        }
        CF_UNPACR_B;
        TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 1, THCON_SEC1_REG3_Base_cntx1_address_ADDR32);
        TTI_NOP;
        for (uint32_t c = 0; c < ct; ++c) {
            CF_UNPACR_A;
            TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
            TTI_NOP;
        }
    });
}

template <uint32_t kt, uint32_t ct>
ALWI void cf_unpack_acquire(uint32_t x_addr, uint32_t w_addr) {
    constexpr uint32_t len = 3 + 3 * ct;
    volatile uint32_t* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    // SEC0 (srcA) = in1 = w, SEC1 (srcB) = in0 = x.
    _llk_unpack_configure_addresses_(w_addr, x_addr, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    if (unp_cfg_context == 0) {
#pragma GCC unroll 16
        for (uint32_t kk = 0; kk < kt; ++kk) {
            TTI_REPLAY(0, len, 0, 0);
        }
    } else {
#pragma GCC unroll 16
        for (uint32_t kk = 0; kk < kt; ++kk) {
            TTI_REPLAY(len, len, 0, 0);
        }
    }
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}

template <uint32_t kt, uint32_t ct>
ALWI void cf_unpack_subblock(uint32_t& x_addr, uint32_t x_step, uint32_t w_addr, uint32_t w_step) {
#ifdef CF_UNROLL
#pragma GCC unroll 16
#endif
    for (uint32_t kk = 0; kk < kt; ++kk) {
        _llk_unpack_AB_matmul_(x_addr, w_addr, 0, 0, x_step, w_step, false, false, ct, 1, kt);
        x_addr += x_step;
        w_addr += ct * w_step;
    }
}
#endif

template <
    uint32_t k_block_tiles,
    uint32_t num_k_rounds,
    uint32_t core_n_tiles,
    uint32_t core_row_tiles_max,
    typename Buf>
ALWI void floor_accumulate_wave(
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
    constexpr uint32_t w_chunk_tiles = k_block_tiles * core_n_tiles;
    constexpr uint32_t x_round_tiles = core_row_tiles_max * k_block_tiles;
#ifdef CF_ROWS
    constexpr uint32_t acq_rows = CF_ROWS;
#else
    constexpr uint32_t acq_rows = compute_kernel_lib::DEST_AUTO_LIMIT / core_n_tiles;
#endif
    static_assert(
        acq_rows >= 1 && acq_rows * core_n_tiles <= compute_kernel_lib::DEST_AUTO_LIMIT,
        "one tile row must fit a DEST half");

    ckernel::reconfig_data_format(cb_w, cb_x);
    ckernel::pack_reconfig_data_format(cb_partials);
    ckernel::matmul_block_init(cb_x, cb_w, true, core_n_tiles, 1, k_block_tiles);
#ifdef CF_DEST_RWC
    MATH((ckernel::addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 1},
        .dest = {.incr = 64, .clr = 0, .cr = 1},
    }
              .set(ADDR_MOD_3)));
#endif
#ifdef CF_UNPACK_REPLAY
    UNPACK(({
        cf_unpack_replay_load<core_n_tiles>();
        volatile uint32_t* cfg = get_cfg_pointer();
        cfg[SCRATCH_SEC0_val_ADDR32] = get_local_cb_interface(cb_w).fifo_page_size;
        cfg[SCRATCH_SEC1_val_ADDR32] = get_local_cb_interface(cb_x).fifo_page_size;
    }));
#endif

    MaybeDeviceZoneScope("compute_accumulate");
    for (uint32_t round = 0; round < num_k_rounds; ++round) {
        if (my_rows > 0) {
            MaybeDeviceZoneScope("compute_wait_inputs");
            x_buf.wait_front(my_rows * k_block_tiles);
            w_buf.wait_front(last_slot * w_chunk_tiles);
        }
        pack_reconfig_l1_acc(round > 0 ? 1 : 0);

        // Unpacker L1 addresses (16 B units, rd_ptr - 1 as in llk_unpack_AB_matmul).
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
            const uint32_t w_expert = w_base + e * w_chunk_tiles * w_step;
            uint32_t out_row = (first - row_lo) * core_n_tiles;
            for (uint32_t r = 0; r < rows; r += acq_rows) {
                const uint32_t n_rows = rows - r < acq_rows ? rows - r : acq_rows;
                tile_regs_acquire();
                for (uint32_t rr = 0; rr < n_rows; ++rr) {
                    MATH((cf_math_subblock<k_block_tiles, core_n_tiles>(cb_x, cb_w, rr * core_n_tiles)));
#ifdef CF_UNPACK_REPLAY
                    UNPACK((cf_unpack_acquire<k_block_tiles, core_n_tiles>(x_addr, w_expert)));
                    UNPACK((x_addr += k_block_tiles * x_step));
#else
                    UNPACK((cf_unpack_subblock<k_block_tiles, core_n_tiles>(x_addr, x_step, w_expert, w_step)));
#endif
                }
                tile_regs_commit();
                tile_regs_wait();
#ifndef CF_DIAG_NO_PACK
                for (uint32_t t = 0; t < n_rows * core_n_tiles; ++t) {
                    pack_tile<true>(t, cb_partials, out_row + t);
                }
#endif
                tile_regs_release();
                out_row += n_rows * core_n_tiles;
            }
        }
        x_buf.wait_front(x_round_tiles);
        x_buf.pop_front(x_round_tiles);
        w_buf.wait_front(MOE_NUM_EXPERTS * w_chunk_tiles);
        w_buf.pop_front(MOE_NUM_EXPERTS * w_chunk_tiles);
    }
}
