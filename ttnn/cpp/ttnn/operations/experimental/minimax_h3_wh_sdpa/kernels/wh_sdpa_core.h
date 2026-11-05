// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Ported verbatim from tt-lab-private h3-wh src/wh_sdpa_trisc.h (includes adapted to tt-metal).
// The WH SDPA core loop on one Tensix tile, shared by the single-tile bench (trisc_wh_sdpa.cpp), the DiT op
// (trisc_wh_sdpa_op.cpp) and the video VAE's attention (trisc_wh_vae_attn.cpp): the device side of whp_sdpa_step
// (wh_proxy.h), one 64-query block of a head (WhSdpaHead: 128 or 64, a template argument) against 64-key blocks. Key
// block j lives in S^T buffer j % 2 (Dst half 1), O^T in Dst half 0, both slice major (wh_fill_row); a head of 64 has
// one O^T d half. TRISC1 runs QK(0), then per block QK(j + 1) and PV(j): QK fills the buffer with S^T = K Q^T; PV runs
// O^T += V^T P^T per key slice, SrcA unpacked from P^T in L1. TRISC0 unpacks in that order. TRISC2 runs block j's
// softmax under PV(j - 1): the block maximum paired vector by vector with block j - 1's row sums (P(j - 1), packed,
// in the other buffer), l of j - 1, the new m and the exp in place, which leaves P's BF16 bits raw in Dst. Then it
// packs P(j) to L1 buffer j % 2 as opaque 16 bits before QK(j + 1) starts (the packer shares a Dst half with neither
// the SFPU nor the FPU), computes alpha in LRegs under the pack and scales O^T by alpha under QK(j + 1). At the end
// it sums the last P and divides O^T by l.
//
// Thread 2 keeps a query's records m and l as one BF16 pair (m high) in "compact" LRegs: LReg 6 + p holds the queries
// of column parity p, lane (r, k) query 16 r + 2 k + p. Per-face vectors (an SFPLOAD's 4 keys by 8 queries) reduce
// to compact form through SFPTRANSP pairs, which leave LRegs 4-7 as they were, and a per-row pick. Compact values
// reach a face's lanes by SFPCONFIG's broadcast of LReg 0's first row into LRegs 11-14.
//
// L1 operands (chunks of WH_SDPA_FACE_BYTES): unpacker 0 reads chunk W x 8 + Z, Q^T at W 0 (d slice f's chunk
// [query face c][d][query]) and P^T's two buffers at W 1 (buffer b's key slice s, [query face c][16 keys][query], at Z
// 4 b + s); unpacker 1 reads chunk W x whp_sdpa_block_chunks + Z of the K/V blocks, block W's K d slice f ([key][d]) at
// Z f and V^T's key slice s and d half h ([d][key]) at Z slices + halves s + h.
#pragma once

#include "wh_sdpa_consts.h"
#include "wh_fpu.h"

#ifndef TT_TRISC_INDEX
#error "wh_sdpa_core.h is a compute kernel header (UCK_CHLKC_UNPACK / MATH / PACK)"
#endif

// Thread 1 posts QK(j) and PV(j) done; thread 2 posts block j's softmax and pack done (its last access to Dst half 1),
// P(j) ready (O^T rescaled) to thread 1 and P(j) packed to thread 0. The op's transposes add thread 1's moves done (to
// thread 2) and Q^T packed (to thread 0).
static constexpr uint32_t sem_qk = 1u << 1, sem_p = 1u << 2, sem_pv = 1u << 3, sem_softmax = 1u << 4,
                          sem_packed = 1u << 5, sem_moved = 1u << 6, sem_qt = 1u << 7;

// Thread 2 must not touch the Dst half the FPU is using (notes/minimax-h3-wh-compute.md): O^T fills half 0, the S^T
// buffers half 1.
static constexpr uint32_t s_base(uint32_t b) { return 512u + 256u * b; }

static constexpr uint32_t o_base(uint32_t h) { return 256u * h; }

#if TT_TRISC_INDEX == 0

// Unpacker 0's region at q_base (Q^T at W 0, P^T at W 1), unpacker 1's at kv_base.
template <const WhSdpaHead& head>
static void setup_unpack(uint32_t q_base, uint32_t kv_base) {
    wh_unpack_setup(q_base, 8u, 2u, kv_base, whp_sdpa_block_chunks(head), 2u);
}

// QK's operands: Q^T's d slices and K's of block W `kv`.
template <const WhSdpaHead& head>
static void unpack_qk(uint32_t kv) {
    wh_push(TT_OP_SETADC(wh_adc_unpack0, 0u, wh_adc_w, 0u));
    wh_push(TT_OP_SETADC(wh_adc_unpack1, 0u, wh_adc_w, kv));
    wh_push(TT_OP_SETADC(wh_adc_unpack0 | wh_adc_unpack1, 0u, wh_adc_z, 0u));
    for (uint32_t f = 0; f < head.slices; ++f) {
        wh_push(wh_unpack_a);
        wh_push(wh_unpack_b);
    }
}

// PV's operands once thread 2 has packed P^T into its buffer `p` (j % 2): P^T's key slice s and V^T's d halves of
// block W `kv`.
template <const WhSdpaHead& head>
static void unpack_pv(uint32_t p, uint32_t kv) {
    wh_push(TT_OP_SEMWAIT(wh_block_sync | wh_block_unpack, sem_packed, wh_sem_on_zero));
    wh_push(TT_OP_SEMGET(sem_packed));
    wh_push(TT_OP_SETADC(wh_adc_unpack0, 0u, wh_adc_w, 1u));
    wh_push(TT_OP_SETADC(wh_adc_unpack0, 0u, wh_adc_z, 4u * p));
    wh_push(TT_OP_SETADC(wh_adc_unpack1, 0u, wh_adc_w, kv));
    wh_push(TT_OP_SETADC(wh_adc_unpack1, 0u, wh_adc_z, head.slices));
    for (uint32_t s = 0; s < 4u; ++s) {
        wh_push(wh_unpack_a);
        for (uint32_t h = 0; h < head.halves; ++h) {
            wh_push(wh_unpack_b);
        }
    }
}

static constexpr uint32_t gpr_unpack_cfg = 30u;

// Unpacker 0 transposing each 16 x 16 face (Haloize), or not; throttled as wh_unpack_setup leaves it.
[[maybe_unused]] static void transpose_faces(bool on) {
    wh_push(TT_OP_STALLWAIT(wh_block_cfg | wh_block_thcon, wh_wait_unpack0));
    wh_write_cfg(
        gpr_unpack_cfg,
        THCON_SEC0_REG2_Out_data_format_ADDR32,
        wh_fmt_bf16 | 2u << 4 | (on ? THCON_SEC0_REG2_Haloize_mode_MASK : 0u));
}

// `chunks` chunks of unpacker 0's group W `w` to SrcA in turn, every face transposed (transpose_faces on), for
// TRISC1's moves.
[[maybe_unused]] static void unpack_chunks(uint32_t w, uint32_t chunks) {
    wh_push(TT_OP_SETADC(wh_adc_unpack0, 0u, wh_adc_w, w));
    wh_push(TT_OP_SETADC(wh_adc_unpack0, 0u, wh_adc_z, 0u));
    for (uint32_t f = 0; f < chunks; ++f) {
        wh_push(wh_unpack_a);
    }
}

[[maybe_unused]] static void unpack_transposed(uint32_t w, uint32_t chunks) {
    transpose_faces(true);
    unpack_chunks(w, chunks);
    transpose_faces(false);
}

#elif TT_TRISC_INDEX == 1

static void setup_math() {
    wh_math_setup(true);
    wh_push(TT_OP_SEMINIT(4u, 0u, sem_qk | sem_p | sem_pv | sem_softmax | sem_packed | sem_moved | sem_qt));
}

static void set_dst_offset(uint32_t row) { wh_push(TT_OP_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, row)); }

static void post_when_done(uint32_t sem) {
    wh_push(TT_OP_STALLWAIT(wh_block_sync, wh_wait_math));
    wh_push(TT_OP_SEMPOST(sem));
}

static constexpr uint32_t gpr_math_cfg = 30u;

// `chunks` chunks TRISC0 unpacks transposed, moved from SrcA into Dst past `base` 8 rows at a time (chunk k's group i
// to row(k, i)), with Src denormal flushing off so that the moves copy every datum; then TRISC2 packs them.
template <uint32_t (*row)(uint32_t, uint32_t)>
[[maybe_unused]] static void move_chunks(uint32_t base, uint32_t chunks) {
    wh_push(TT_OP_STALLWAIT(wh_block_cfg, wh_wait_math));
    wh_write_cfg(gpr_math_cfg, ALU_ACC_CTRL_Zero_Flag_disabled_src_ADDR32, 1u);
    set_dst_offset(base);
    for (uint32_t k = 0; k < chunks; ++k) {
        for (uint32_t i = 0; i < 8u; ++i) {
            wh_push(TT_OP_MOVA2D(0u, 8u * i, 0u, 2u, row(k, i)));
        }
        wh_push(TT_OP_SETRWC(1u, 0u, 0u, 0u, 0u, 3u));
    }
    wh_push(TT_OP_STALLWAIT(wh_block_cfg, wh_wait_math));
    wh_write_cfg(gpr_math_cfg, ALU_ACC_CTRL_Zero_Flag_disabled_src_ADDR32, 0u);
    post_when_done(sem_moved);
}

// S^T of block j into its zeroed buffer (j % 2), after thread 2 has left Dst half 1 (`wait`: block j - 1's softmax).
// The first fill (`record`) records the fill's replay.
template <const WhSdpaHead& head>
static void qk(uint32_t buffer, bool wait, bool record, uint32_t phases) {
    if (wait) {
        wh_push(TT_OP_SEMWAIT(wh_block_sync | wh_block_math, sem_softmax, wh_sem_on_zero));
        wh_push(TT_OP_SEMGET(sem_softmax));
    }
    set_dst_offset(s_base(buffer));
    for (uint32_t i = 0; i < 16u; ++i) {
        wh_push(TT_OP_ZEROACC(wh_zeroacc_16_rows_16b, 3u, s_base(buffer) / 16u + i));
    }
    for (uint32_t f = 0; f < head.slices; ++f) {
        wh_fill(phases, record && f == 0u, wh_flip_ab, WhFillRows::slice);
    }
    post_when_done(sem_qk);
}

// O^T += V^T P^T of block j once thread 2 has handed P(j) over, SrcA from P^T's key slices in L1 (kept over a key
// slice's d halves); O^T zeroed first for a query block's first (`zero`).
template <const WhSdpaHead& head>
static void pv(uint32_t phases, bool zero) {
    wh_push(TT_OP_SEMWAIT(wh_block_sync | wh_block_math, sem_p, wh_sem_on_zero));
    wh_push(TT_OP_SEMGET(sem_p));
    for (uint32_t i = 0; i < 16u * head.halves && zero; ++i) {
        wh_push(TT_OP_ZEROACC(wh_zeroacc_16_rows_16b, 3u, o_base(0u) / 16u + i));
    }
    for (uint32_t s = 0; s < 4u; ++s) {
        for (uint32_t h = 0; h < head.halves; ++h) {
            set_dst_offset(o_base(h));
            wh_fill(phases, false, h + 1u < head.halves ? wh_flip_b : wh_flip_ab, WhFillRows::slice);
        }
    }
    post_when_done(sem_pv);
}

#else
static constexpr uint32_t gpr_out = 4u, gpr_cfg = 6u;

// SFPLOAD and SFPSTORE formats: BF16, UINT16 (the low 16 bits raw), a datum raw to or from the high half (the other
// half cleared on a load) and to or from the low half (the other half kept on a load). LRegs 9 (0), 10 (1.0) and 15
// (lane i: 2 i).
static constexpr uint32_t bf16 = 2u, uint16 = 6u, high16 = 7u, low16_only = 14u, high16_only = 15u;
static constexpr uint32_t lreg_zero = 9u, lreg_one = 10u, lreg_lane = 15u;
// SFPLOADI modes: a BF16 to the high half (the low cleared), an unsigned 16 bits (the high cleared), one half kept.
static constexpr uint32_t loadi_bf16 = 0u, loadi_ushort = 2u, loadi_high = 8u, loadi_low = 10u;
// SFPSWAP: VD the maximum, VC the minimum.
static constexpr uint32_t swap_max = 9u;
// The packer's 16-bit integer format: P's raw bits from the exp go to L1 unchanged.
static constexpr uint32_t fmt_int16 = 9u;

// SFPLOADMACRO sequences: 0 the maximum into LReg 0 (SFPSWAP), 1 and 2 the row sums' ping-pong (MAD into LReg 2 from
// LReg 3, and back), 3 the exp. The scaling of O^T takes sequence 1 while it runs.
static constexpr uint32_t macro_max = 0u, macro_sum_a = 1u, macro_sum_b = 2u, macro_exp = 3u, macro_scale = 1u;
static constexpr uint32_t seq_max = 0x00000284u, seq_sum_a = 0x00008500u, seq_sum_b = 0x00008600u;
// The exp: MAD of template 1 at 0, round to UINT16 (template 2) at 2, NOT (template 3) into LReg 16 at 3, store LReg 16
// at 4 in UINT16. The scaling: MAD of template 1 at 0, round to BF16 (template 2) into LReg 16 at 2, store LReg 16 at
// 3 in the load's format.
static constexpr uint32_t seq_exp = 0x6316855Fu, seq_scale = 0x5B568500u;
// Macro store format UINT16, sequence 1 storing in its load's format.
static constexpr uint32_t macro_misc = uint16 | 2u << 4;

// Replay slots: the maximum and sums body (8 vector pairs, recorded each block for its buffers), the exp body and the
// scaling body (8 vectors each, recorded once).
static constexpr uint32_t replay_pairs = 0u, replay_exp = 16u, replay_scale = 24u;

// The vector of query face c, keys (or d) 4 i to 4 i + 3 and column parity p of a slice-major region, past thread 2's
// Dst offset (the region's base).
static constexpr uint32_t s_vector(uint32_t c, uint32_t i, uint32_t p) {
    return 64u * (i / 4u) + 16u * c + 4u * (i % 4u) + 2u * p;
}

// Replayed face loops address vector i of a face as s_vector(0, i, 0) past a Dst offset at the face, one replay per
// half face (vectors 8 to 15 sit 128 rows past 0 to 7). They cannot step a Dst RWC instead: thread 2's AddrMod
// increments displace the matrix unit's concurrent Dst writes.
static constexpr uint32_t face_vector(uint32_t i) { return s_vector(0u, i, 0u); }

static constexpr uint32_t half_face = 128u;

// Scratch in the buffer the sums read, past scratch_base: face (c, p)'s vectors 8 on, free once its sums are read.
// Slot 0 the face maximum, 1 and 2 the face sums' high and low halves, 3 and 4 the records' halves (face 0) or a
// parity 0 reduction kept for later (face (1, 0)), 5 and 6 the exp's bias.
static constexpr uint32_t scratch_base(uint32_t buffer) { return s_base(buffer) + half_face; }

static constexpr uint32_t scratch(uint32_t c, uint32_t p, uint32_t slot) {
    return s_vector(c, 8u + slot, p) - half_face;
}

// A Dst address past the offset, modulo the 1024 rows the SFPU's address wraps at.
static constexpr uint32_t dst_wrap(uint32_t rows) { return rows & 1023u; }

static constexpr uint32_t slot_max = 0u, slot_sum = 1u, slot_record = 3u, slot_bias = 5u;

[[gnu::always_inline]] static inline void push(uint32_t instruction) { wh_push(instruction); }

[[gnu::always_inline]] static inline void nop() { push(TT_OP_SFPNOP); }

[[gnu::always_inline]] static inline void load(uint32_t lreg, uint32_t address) {
    push(TT_OP_SFPLOAD(lreg, bf16, 0u, address));
}

[[gnu::always_inline]] static inline void store(uint32_t lreg, uint32_t address) {
    push(TT_OP_SFPSTORE(lreg, bf16, 0u, address));
}

// A 32-bit value through two 16-bit Dst datums: the high half at vector `high`, the low half at `low`.
[[gnu::always_inline]] static inline void store_wide(uint32_t lreg, uint32_t high, uint32_t low) {
    push(TT_OP_SFPSTORE(lreg, high16_only, 0u, high));
    push(TT_OP_SFPSTORE(lreg, low16_only, 0u, low));
}

[[gnu::always_inline]] static inline void load_wide(uint32_t lreg, uint32_t high, uint32_t low) {
    push(TT_OP_SFPLOAD(lreg, high16, 0u, high));
    push(TT_OP_SFPLOAD(lreg, low16_only, 0u, low));
}

// x y + z into d.
[[gnu::always_inline]] static inline void mad(uint32_t x, uint32_t y, uint32_t z, uint32_t d) {
    push(TT_OP_SFPMAD(x, y, z, d, 0u));
}

[[gnu::always_inline]] static inline void mov(uint32_t from, uint32_t to) { push(TT_OP_SFPMOV(0u, from, to, 0u)); }

// d = d & c, d = d | c.
[[gnu::always_inline]] static inline void and_into(uint32_t c, uint32_t d) { push(TT_OP_SFPAND(0u, c, d, 0u)); }

[[gnu::always_inline]] static inline void or_into(uint32_t c, uint32_t d) { push(TT_OP_SFPOR(0u, c, d, 0u)); }

// d shifted left by `shift` bits (right when negative).
[[gnu::always_inline]] static inline void shift(uint32_t d, int32_t shift_bits) {
    push(TT_OP_SFPSHFT(uint32_t(shift_bits) & 0xFFFu, 0u, d, 1u));
}

[[gnu::always_inline]] static inline void max_into(uint32_t c, uint32_t d) {
    push(TT_OP_SFPSWAP(0u, c, d, swap_max));
    nop();
}

[[gnu::always_inline]] static inline void round_bf16(uint32_t c, uint32_t d) {
    push(TT_OP_SFP_STOCH_RND(0u, 0u, 0u, c, d, 1u));
}

[[gnu::always_inline]] static inline void lanes_masked(bool on) { push(TT_OP_SFPENCC(on ? 1u : 0u, 0u, 0u, 2u)); }

[[gnu::always_inline]] static inline void config_from_lreg0(uint32_t dest) { push(TT_OP_SFPCONFIG(0u, dest, 0u)); }

[[gnu::always_inline]] static inline void transpose() { push(TT_OP_SFPTRANSP(0u, 0u, 0u, 0u)); }

// A replay of `slots` slots from `start`.
[[gnu::always_inline]] static inline void replay(uint32_t start, uint32_t slots) {
    push(TT_OP_REPLAY(start, slots, 0u, 0u));
}

[[gnu::always_inline]] static inline void load_constant(uint32_t lreg, uint32_t bits) {
    if (bits & 0xFFFFu) {
        push(TT_OP_SFPLOADI(lreg, loadi_low, bits & 0xFFFFu));
        push(TT_OP_SFPLOADI(lreg, loadi_high, bits >> 16));
    } else {
        push(TT_OP_SFPLOADI(lreg, loadi_bf16, bits >> 16));
    }
}

// The high half's mask (0xFFFF0000) or the low half's.
[[gnu::always_inline]] static inline void load_high_mask(uint32_t lreg) {
    push(TT_OP_SFPLOADI(lreg, loadi_bf16, 0xFFFFu));
}

[[gnu::always_inline]] static inline void load_low_mask(uint32_t lreg) {
    push(TT_OP_SFPLOADI(lreg, loadi_ushort, 0xFFFFu));
}

// Sequence `macro`: a 16-bit value from the immediate, a wider one through LReg 0.
[[gnu::always_inline]] static inline void sequence(uint32_t macro, uint32_t bits) {
    if (bits >> 16) {
        load_constant(0u, bits);
        config_from_lreg0(4u + macro);
    } else {
        push(TT_OP_SFPCONFIG(bits, 4u + macro, 1u));
    }
}

// Instruction templates by backdoor loads (VD 12 + slot). Template 0 is the maximum's SFPSWAP into LReg 0 (VC), 3 the
// exp's NOT; templates 1 and 2 follow the pass.
[[gnu::always_inline]] static inline void templates_sums() {
    push(TT_OP_SFPMAD(lreg_one, 0u, 3u, 13u, 0u));
    push(TT_OP_SFPMAD(lreg_one, 0u, 2u, 14u, 0u));
}

// The exp: -A (LReg 5) x + the face's bias (LReg 6), then to UINT16.
[[gnu::always_inline]] static inline void templates_exp() {
    push(TT_OP_SFPMAD(5u, 0u, 6u, 13u, 0u));
    push(TT_OP_SFP_STOCH_RND(0u, 0u, 0u, 0u, 14u, 6u));
}

// The scaling: the face's factor (LReg 5) x, then to BF16.
[[gnu::always_inline]] static inline void templates_scale() {
    push(TT_OP_SFPMAD(5u, 0u, lreg_zero, 13u, 0u));
    push(TT_OP_SFP_STOCH_RND(0u, 0u, 0u, 0u, 14u, 1u));
}

// The exp and scaling face bodies: 8 vectors each, loads rotating over LRegs 0-3.
static void record_face_bodies() {
    push(TT_OP_REPLAY(replay_exp, 8u, 0u, 1u));
#pragma GCC unroll 16
    for (uint32_t i = 0; i < 8u; ++i) {
        push(TT_OP_SFPLOADMACRO(macro_exp << 2 | i % 4u, bf16, 0u, face_vector(i)));
    }
    push(TT_OP_REPLAY(replay_scale, 8u, 0u, 1u));
#pragma GCC unroll 16
    for (uint32_t i = 0; i < 8u; ++i) {
        push(TT_OP_SFPLOADMACRO(macro_scale << 2 | i % 4u, bf16, 0u, face_vector(i)));
    }
}

static void setup_macros() {
    push(TT_OP_SFPSWAP(0u, 0u, 12u, 1u));
    push(TT_OP_SFPNOT(0u, 0u, 15u, 0u));
    sequence(macro_max, seq_max);
    sequence(macro_sum_b, seq_sum_b);
    sequence(macro_exp, seq_exp);
    push(TT_OP_SFPCONFIG(macro_misc, 8u, 1u));
    record_face_bodies();
}

// A query block's records before its first key block: m the lowest finite BF16, l 0.
[[gnu::always_inline]] static inline void reset_records() {
#pragma GCC unroll 16
    for (uint32_t p = 0; p < 2u; ++p) {
        load_constant(6u + p, whp_sdpa_m_init);
    }
}

static void setup_softmax() {
    wh_pack_setup(true);
    wh_math_addr_mod(0u, 0u, 0u);
    setup_macros();
}

// The SFPU's Dst offset, once its earlier instructions are done.
[[gnu::always_inline]] static inline void set_dst_offset(uint32_t row) {
    push(TT_OP_STALLWAIT(wh_block_cfg, wh_wait_sfpu));
    push(TT_OP_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, row));
}

// The SFPU's Dst offset for later instructions: in-order issue keeps earlier ones on the old offset.
[[gnu::always_inline]] static inline void move_dst_offset(uint32_t row) {
    push(TT_OP_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, row));
}

[[gnu::always_inline]] static inline void post_when_done(uint32_t sem) {
    push(TT_OP_STALLWAIT(wh_block_sync, wh_wait_sfpu));
    push(TT_OP_SEMPOST(sem));
}

[[gnu::always_inline]] static inline void wait_and_take(uint32_t sem) {
    push(TT_OP_SEMWAIT(wh_block_sync | wh_block_sfpu, sem, wh_sem_on_zero));
    push(TT_OP_SEMGET(sem));
}

// `regions` regions of 256 Dst rows from `rows` packed to L1 `addr` (region h at + h x WH_SDPA_REGION_BYTES, chunks of
// 64 rows) once TRISC1's moves into them are done.
[[maybe_unused]] static void pack_moved(uint32_t rows, uint32_t addr, uint32_t regions) {
    push(TT_OP_SEMWAIT(
        wh_block_sync | wh_block_pack | wh_block_cfg | wh_block_thcon | wh_block_sfpu, sem_moved, wh_sem_on_zero));
    push(TT_OP_SEMGET(sem_moved));
    for (uint32_t h = 0; h < regions; ++h) {
        wh_pack_region(gpr_out, addr + h * WH_SDPA_REGION_BYTES, WH_SDPA_FACE_BYTES, rows / 16u + 16u * h);
    }
    push(TT_OP_STALLWAIT(wh_block_sync | wh_block_cfg | wh_block_thcon, wh_wait_pack));
}

// Packs from the 16-bit Dst as opaque 16 bits (P's raw bits) or as BF16. Only the packers read the Dst format
// override.
[[gnu::always_inline]] static inline void pack_format(bool raw) {
    const uint32_t format = raw ? fmt_int16 : wh_fmt_bf16;
    const uint32_t dst =
        raw ? ALU_FORMAT_SPEC_REG_Dstacc_override_MASK | format << ALU_FORMAT_SPEC_REG_Dstacc_val_SHAMT : 0u;
    wh_write_cfg(gpr_cfg, ALU_FORMAT_SPEC_REG_Dstacc_val_ADDR32, dst);
    wh_set_packer_cfg(gpr_cfg, 1u, 1u | format << 4 | format << 8, 0u);
}

// LRegs 0-3, a face each (rows: keys 4 i + r), to the compact maxima or sums over rows in LReg 0. The transpose leaves
// LRegs 4-7 transposed until the next reduction transposes them back, so reductions come in pairs.
[[gnu::always_inline]] static inline void reduce_faces(bool maximum) {
    nop();
    transpose();
    if (maximum) {
        max_into(1u, 0u);
        max_into(3u, 2u);
        max_into(2u, 0u);
    } else {
        mad(0u, lreg_one, 1u, 0u);
        mad(2u, lreg_one, 3u, 2u);
        nop();
        mad(0u, lreg_one, 2u, 0u);
        nop();
    }
}

// The scratch slot of parity p's four faces into LRegs 0-3 (Dst offset at the scratch base).
[[gnu::always_inline]] static inline void load_faces(uint32_t p, uint32_t slot, bool wide) {
#pragma GCC unroll 16
    for (uint32_t c = 0; c < 4u; ++c) {
        if (wide) {
            load_wide(c, scratch(c, p, slot), scratch(c, p, slot + 1u));
        } else {
            push(TT_OP_SFPLOAD(c, high16, 0u, scratch(c, p, slot)));
        }
    }
}

// Body of the maximum and sums pass: the maximum of 8 vectors at `max_rows` past the face (macro 0, loads into LReg
// 1), each paired with the next of the row sums of P at `sum_rows` (macros 1 and 2 in turn, loads into LRegs 2 and 3).
// The SFPSWAP's MAD slot and the cycle after it stay free of the sums' MADs.
[[gnu::always_inline]] static inline void record_pairs(uint32_t max_rows, uint32_t sum_rows) {
    push(TT_OP_REPLAY(replay_pairs, 16u, 0u, 1u));
#pragma GCC unroll 16
    for (uint32_t i = 0; i < 8u; ++i) {
        push(TT_OP_SFPLOADMACRO(macro_max << 2 | 1u, bf16, 0u, max_rows + face_vector(i)));
        const uint32_t macro = i % 2u ? macro_sum_b << 2 | 3u : macro_sum_a << 2 | 2u;
        push(TT_OP_SFPLOADMACRO(macro, high16, 0u, sum_rows + face_vector(i)));
    }
}

// The maxima and sums pass's templates, sequence and replay for block j's maxima (S^T buffer `max_buffer`) and block
// j - 1's row sums (P in buffer `sum_buffer`). They touch no Dst, so they run while the FPU holds Dst half 1.
[[gnu::always_inline]] static inline void pairs_setup(uint32_t max_buffer, uint32_t sum_buffer) {
    templates_sums();
    sequence(macro_sum_a, seq_sum_a);
    record_pairs(s_base(max_buffer) - s_base(0u), s_base(sum_buffer) - s_base(0u));
}

// The maxima and row sums pairs_setup recorded (P in buffer `sum_buffer`, already packed), face by face: per face the
// maximum into LReg 0 from -inf and one sum chain per key row into LReg 3 from 0 (the proxy's order), then both into
// the face's scratch.
[[gnu::always_inline]] static inline void maxima_and_sums(uint32_t sum_buffer) {
    const uint32_t scratch_rows = s_base(sum_buffer) - s_base(0u);
#pragma GCC unroll 16
    for (uint32_t f = 0; f < 8u; ++f) {
        const uint32_t c = f / 2u, p = f % 2u;
        move_dst_offset(s_base(0u) + s_vector(c, 0u, p));
        push(TT_OP_SFPLOADI(0u, loadi_bf16, 0xFF80u));
        mov(lreg_zero, 3u);
        replay(replay_pairs, 16u);
        move_dst_offset(s_base(0u) + s_vector(c, 0u, p) + half_face);
        replay(replay_pairs, 16u);
        nop();
        push(TT_OP_SFPSTORE(0u, high16_only, 0u, scratch_rows + scratch(0u, 0u, slot_max)));
        store_wide(3u, scratch_rows + scratch(0u, 0u, slot_sum), scratch_rows + scratch(0u, 0u, slot_sum + 1u));
    }
}

// The new m of parity p from its block maximum in LReg 0 (whp_sdpa_step's softmax_stats) into its record, with -1 in
// LReg 5. The old m goes to LReg 4 for alphas: parity 1 first, to the low half, then parity 0 to the high half.
template <const WhSdpaHead& head>
[[gnu::always_inline]] static inline void stats_m(uint32_t p) {
    const uint32_t record = 6u + p;
    load_high_mask(1u);
    mov(record, 2u);
    and_into(1u, 2u);     // m_old
    mad(2u, 5u, 0u, 3u);  // block max - m_old
    nop();
    push(TT_OP_SFPADDI(head.margin >> 16, 3u, 0u));
    nop();
    mov(2u, 1u);
    lanes_masked(true);
    push(TT_OP_SFPSETCC(0u, 3u, 0u, 4u));
    push(TT_OP_SFPSETCC(0u, 3u, 0u, 2u));
    mov(0u, 1u);  // m_new
    lanes_masked(false);
    load_low_mask(3u);
    and_into(3u, record);
    or_into(1u, record);
    if (p) {
        shift(2u, -16);
        mov(2u, 4u);
    } else {
        or_into(2u, 4u);
    }
}

// Alpha of parity p into LReg `out` from its old m in LReg 2 and the new m in its record: the rest of softmax_stats.
template <const WhSdpaHead& head>
[[gnu::always_inline]] static inline void alpha(uint32_t p, uint32_t out) {
    load_constant(out, 0xBF800000u);
    load_high_mask(1u);
    mov(6u + p, 3u);
    and_into(1u, 3u);      // m_new
    mad(3u, out, 2u, 3u);  // m_old - m_new
    load_constant(0u, head.log2_scale);
    mad(3u, 0u, lreg_zero, 3u);
    load_constant(0u, whp_sdpa_x_min);
    max_into(0u, 3u);  // x
    load_constant(0u, whp_sdpa_magic);
    mad(3u, lreg_one, 0u, 1u);  // x + magic
    nop();
    mad(0u, out, 1u, 2u);  // round(x)
    nop();
    mad(2u, out, 3u, 3u);  // f
    load_constant(0u, whp_sdpa_poly[2]);
    load_constant(2u, whp_sdpa_poly[1]);
    mad(0u, 3u, 2u, 0u);
    load_constant(2u, whp_sdpa_poly[0]);
    mad(0u, 3u, 2u, 0u);
    nop();
    mad(0u, 3u, lreg_one, 0u);
    shift(1u, 23);
    mad(0u, 1u, lreg_zero, 0u);
    nop();
    round_bf16(0u, out);
}

// Both parities' alpha, packed into LReg 4 (parity 0 high), from the old m stats_m left there. It touches no Dst, so
// it runs under the pack of P.
template <const WhSdpaHead& head>
[[gnu::always_inline]] static inline void alphas() {
    mov(4u, 2u);
    shift(2u, 16);
    alpha<head>(1u, 5u);
    load_high_mask(2u);
    and_into(4u, 2u);
    alpha<head>(0u, 4u);
    shift(5u, -16);
    or_into(5u, 4u);
}

// Alpha of parity p, packed in LReg 4 (parity 0 high), into LReg `to`, through `scratch`.
[[gnu::always_inline]] static inline void unpack_alpha(uint32_t p, uint32_t to, uint32_t scratch_lreg) {
    mov(4u, to);
    if (p) {
        shift(to, 16);
    } else {
        load_high_mask(scratch_lreg);
        and_into(scratch_lreg, to);
    }
}

// l = alpha l + the row sum of parity p in LReg 0, rounded to BF16, into its record, alpha from LReg 4.
[[gnu::always_inline]] static inline void update_l(uint32_t p) {
    const uint32_t record = 6u + p;
    unpack_alpha(p, 1u, 2u);
    mov(record, 2u);
    shift(2u, 16);
    mad(1u, 2u, 0u, 3u);
    nop();
    round_bf16(3u, 3u);
    shift(3u, -16);
    load_high_mask(2u);
    and_into(2u, record);
    or_into(3u, record);
}

// The compact values of LReg 0 as their query faces' vectors in LRegs 11-14 (face c in 11 + c); LRegs 4-7 kept.
[[gnu::always_inline]] static inline void park_faces() {
#pragma GCC unroll 16
    for (uint32_t r = 1; r < 4u; ++r) {
        mov(0u, r);
    }
    transpose();
    nop();
#pragma GCC unroll 16
    for (uint32_t c = 0; c < 4u; ++c) {
        if (c) {
            mov(c, 0u);
            nop();
        }
        config_from_lreg0(11u + c);
    }
    transpose();
}

// Parity p's exp in `buffer`: P's BF16 bits = 65535 less 65535 - t rounded to UINT16 (the clamp), raw. The bias of m
// (from the spilled record) goes to the scratch as face vectors, then per face to LReg 6 by loads, which unlike a move
// do not meet the exp's NOTs on the simple sub-unit.
template <const WhSdpaHead& head, uint32_t buffer>
[[gnu::always_inline]] static inline void exp_parity(uint32_t p) {
    constexpr uint32_t other = 1u - buffer, scratch_row = scratch_base(other);
    move_dst_offset(scratch_row);
    push(TT_OP_SFPLOAD(0u, high16, 0u, scratch(0u, p, slot_record)));
    load_constant(1u, head.exp_a);
    load_constant(2u, whp_sdpa_exp_c);
    mad(0u, 1u, 2u, 0u);
    nop();
#pragma GCC unroll 16
    for (uint32_t r = 1; r < 4u; ++r) {
        mov(0u, r);
    }
    transpose();
#pragma GCC unroll 16
    for (uint32_t c = 0; c < 4u; ++c) {
        store_wide(c, scratch(c, p, slot_bias), scratch(c, p, slot_bias + 1u));
    }
    transpose();
#pragma GCC unroll 16
    for (uint32_t c = 0; c < 4u; ++c) {
        const uint32_t face = s_base(buffer) + s_vector(c, 0u, p);
        move_dst_offset(face);
        load_wide(
            6u,
            dst_wrap(scratch_row + scratch(c, p, slot_bias) - face),
            dst_wrap(scratch_row + scratch(c, p, slot_bias + 1u) - face));
        replay(replay_exp, 8u);
        move_dst_offset(face + half_face);
        replay(replay_exp, 8u);
    }
}

// The exp of block j in `buffer`, the records spilled to the scratch while LRegs 5 and 6 hold -A and the bias.
template <const WhSdpaHead& head, uint32_t buffer>
[[gnu::always_inline]] static inline void exp_block() {
    constexpr uint32_t scratch_row = scratch_base(1u - buffer);
    move_dst_offset(scratch_row);
#pragma GCC unroll 16
    for (uint32_t p = 0; p < 2u; ++p) {
        store_wide(6u + p, scratch(0u, p, slot_record), scratch(0u, p, slot_record + 1u));
    }
    templates_exp();
    load_constant(5u, head.exp_a ^ 0x80000000u);
#pragma GCC unroll 16
    for (uint32_t p = 0; p < 2u; ++p) {
        exp_parity<head, buffer>(p);
    }
    move_dst_offset(scratch_row);
#pragma GCC unroll 16
    for (uint32_t p = 0; p < 2u; ++p) {
        load_wide(6u + p, scratch(0u, p, slot_record), scratch(0u, p, slot_record + 1u));
    }
}

// O^T times the compact factor of parity p in LReg 0, face by face (the factor in LReg 5), every d half.
template <const WhSdpaHead& head>
[[gnu::always_inline]] static inline void scale_parity(uint32_t p) {
    park_faces();
#pragma GCC unroll 16
    for (uint32_t c = 0; c < 4u; ++c) {
        mov(11u + c, 5u);
#pragma GCC unroll 16
        for (uint32_t h = 0; h < head.halves; ++h) {
#pragma GCC unroll 16
            for (uint32_t half = 0; half < 2u; ++half) {
                move_dst_offset(o_base(h) + s_vector(c, 0u, p) + half * half_face);
                replay(replay_scale, 8u);
            }
        }
    }
}

[[gnu::always_inline]] static inline void scale_begin() {
    templates_scale();
    sequence(macro_scale, seq_scale);
}

template <const WhSdpaHead& head>
[[gnu::always_inline]] static inline void rescale_o() {
    scale_begin();
#pragma GCC unroll 16
    for (uint32_t p = 0; p < 2u; ++p) {
        unpack_alpha(p, 0u, 1u);
        scale_parity<head>(p);
    }
}

// O^T / l: a Newton reciprocal of l (3 steps from the bit-pattern seed) per parity.
template <const WhSdpaHead& head>
[[gnu::always_inline]] static inline void normalize() {
    scale_begin();
#pragma GCC unroll 16
    for (uint32_t p = 0; p < 2u; ++p) {
        mov(6u + p, 1u);
        shift(1u, 16);
        load_constant(2u, whp_sdpa_recip_seed);
        mov(1u, 3u);
        push(TT_OP_SFPIADD(0u, 2u, 3u, 6u));  // seed - l
        push(TT_OP_SFPMOV(0u, 1u, 1u, 1u));   // -l
        load_constant(2u, 0x40000000u);
#pragma GCC unroll 16
        for (uint32_t step = 0; step < 3u; ++step) {
            mad(1u, 3u, 2u, 0u);
            nop();
            mad(3u, 0u, lreg_zero, 3u);
            nop();
        }
        mov(3u, 0u);
        scale_parity<head>(p);
    }
}

// Block j - 1's l from its row sums in the scratch (alpha of j - 1 in LReg 4), parity 0's compact sums kept in the
// scratch while parity 1's reduction transposes LRegs 4-7 back.
[[gnu::always_inline]] static inline void sums_to_l(uint32_t scratch_row) {
    constexpr uint32_t keep = scratch(1u, 0u, slot_record), keep_low = scratch(1u, 0u, slot_record + 1u);
    move_dst_offset(scratch_row);
    load_faces(0u, slot_sum, true);
    reduce_faces(false);
    store_wide(0u, keep, keep_low);
    load_faces(1u, slot_sum, true);
    reduce_faces(false);
    update_l(1u);
    load_wide(0u, keep, keep_low);
    update_l(0u);
}

// Block j's new m from its face maxima in the scratch, parity 0's compact maximum kept as in sums_to_l, the old m in
// LReg 4.
template <const WhSdpaHead& head>
[[gnu::always_inline]] static inline void maxima_to_m(uint32_t scratch_row) {
    constexpr uint32_t keep = scratch(1u, 0u, slot_record);
    move_dst_offset(scratch_row);
    load_faces(0u, slot_max, false);
    reduce_faces(true);
    push(TT_OP_SFPSTORE(0u, high16_only, 0u, keep));
    load_faces(1u, slot_max, false);
    reduce_faces(true);
    load_constant(5u, 0xBF800000u);
    stats_m<head>(1u);
    push(TT_OP_SFPLOAD(0u, high16, 0u, keep));
    stats_m<head>(0u);
}

// The padding keys (`valid` on) of the S^T buffer at `base` to -inf, so that the exp's saturation zeroes their P; the
// vector holding key `valid` takes it in its rows from valid % 4 on.
static void mask_padding(uint32_t base, uint32_t valid) {
    move_dst_offset(base);
    push(TT_OP_SFPLOADI(1u, loadi_bf16, 0xFF80u));
    if (valid % 4u) {
        lanes_masked(true);
        push(TT_OP_SFPIADD((0u - 16u * (valid % 4u)) & 0xFFFu, lreg_lane, 2u, 9u));
    }
#pragma GCC unroll 16
    for (uint32_t i = valid / 4u; i < 16u; ++i) {
#pragma GCC unroll 16
        for (uint32_t c = 0; c < 8u; ++c) {
            store(1u, s_vector(c / 2u, i, c % 2u));
        }
        if (i == valid / 4u && valid % 4u) {
            lanes_masked(false);
        }
    }
}

// The keys past each query of the S^T buffer at `base` to -inf, the block the queries' own (key k against query k):
// key 4 i + r of lane (r, j) against query 16 c + 2 j + p is masked when 2 j - r + p + 16 c - 4 i < 0, so every lane of
// a key group past its query face, none before it, and on the face's diagonal by the lane's 2 j - r (L3, from LReg 15's
// 16 r + 2 j).
static void mask_causal(uint32_t base) {
    move_dst_offset(base);
    mov(lreg_lane, 3u);
    shift(3u, -4);
    push(TT_OP_SFPLOADI(0u, loadi_ushort, 14u));
    mov(lreg_lane, 2u);
    and_into(0u, 2u);
    push(TT_OP_SFPIADD(0u, 2u, 3u, 6u));  // 2 j - r
    push(TT_OP_SFPLOADI(1u, loadi_bf16, 0xFF80u));
#pragma GCC unroll 1
    for (uint32_t i = 0; i < 16u; ++i) {
#pragma GCC unroll 1
        for (uint32_t c = 0; c < i / 4u; ++c) {
            store(1u, s_vector(c, i, 0u));
            store(1u, s_vector(c, i, 1u));
        }
#pragma GCC unroll 2
        for (uint32_t p = 0; p < 2u; ++p) {
            lanes_masked(true);
            push(TT_OP_SFPIADD((p - 4u * (i % 4u)) & 0xFFFu, 3u, 2u, 1u));
            store(1u, s_vector(i / 4u, i, p));
            lanes_masked(false);
        }
    }
}

// Block j's softmax in `buffer` (j % 2) with block j - 1's row sums and l from the other buffer (P(j - 1), packed): the
// maxima and sums (pairs_setup done), l (`sums`: from block 1 on), the new m (the old in LReg 4) and the exp. With
// `causal` the block is the queries' own (mask_causal).
template <const WhSdpaHead& head, uint32_t buffer>
[[gnu::always_inline]] static inline void softmax(bool sums, uint32_t valid, bool causal) {
    constexpr uint32_t other = 1u - buffer, scratch_row = scratch_base(other);
    if (valid < 64u) {
        mask_padding(s_base(buffer), valid);
    }
    if (causal) {
        mask_causal(s_base(buffer));
    }
    maxima_and_sums(other);
    if (sums) {
        sums_to_l(scratch_row);
    }
    maxima_to_m<head>(scratch_row);
    exp_block<head, buffer>();
}

// Block j (in `buffer`, j % 2): its softmax once QK(j) is done, then P(j) packed to L1 at `p` (buffer j % 2) and, once
// PV(j - 1) is done, O^T scaled by alpha (block 1 on, unless `rescale` is off); `causal` as softmax. QK(j + 1) waits
// for the pack: a pack under its first MVMULs loses some of their writes to query face 0
// (notes/minimax-h3-wh-compute.md). The packers and the maxima pass are set up under QK(j), so only the PACR waits
// for the softmax, and alpha of j runs under the pack.
template <const WhSdpaHead& head, uint32_t buffer>
[[gnu::noinline]] static void block(uint32_t p, bool later, uint32_t valid, bool causal, bool rescale) {
    wh_pack_prepare(gpr_out, p, WH_SDPA_REGION_BYTES / 4u, s_base(buffer) / 16u);
    pairs_setup(buffer, 1u - buffer);
    wait_and_take(sem_qk);
    softmax<head, buffer>(later, valid, causal);
    push(TT_OP_STALLWAIT(wh_block_pack, wh_wait_sfpu));
    wh_pack_prepared();
    push(TT_OP_STALLWAIT(wh_block_sync, wh_wait_pack));
    push(TT_OP_SEMPOST(sem_softmax));
    push(TT_OP_SEMPOST(sem_packed));
    alphas<head>();
    if (later) {
        wait_and_take(sem_pv);
    }
    if (later && rescale) {
        rescale_o<head>();
    }
    post_when_done(sem_p);
}

// The last block's row sums and l, its P in `buffer`.
template <uint32_t buffer>
[[gnu::noinline]] static void last_sums() {
    pairs_setup(1u - buffer, buffer);
    maxima_and_sums(buffer);
    sums_to_l(scratch_base(buffer));
}

// The last block's row sums and l (`blocks` blocks), then O^T / l once PV of the last block is done.
template <const WhSdpaHead& head>
static void finish(uint32_t blocks) {
    if (blocks % 2u) {
        last_sums<0u>();
    } else {
        last_sums<1u>();
    }
    wait_and_take(sem_pv);
    normalize<head>();
}

#endif
