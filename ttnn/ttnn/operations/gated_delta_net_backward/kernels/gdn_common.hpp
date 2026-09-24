// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// gated_delta_net_backward — shared compile-time geometry and tile-address math.
//
// Included by all three kernels.  The compile-time argument block is identical
// across the reader, writer and compute binaries (indices 0..CT_ACC_BASE-1);
// each kernel appends its own TensorAccessorArgs after that.

#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// Compile-time geometry (mirrors `common_ct` in the program descriptor)
// ---------------------------------------------------------------------------
constexpr uint32_t gB = get_compile_time_arg_val(0);
constexpr uint32_t gT = get_compile_time_arg_val(1);
constexpr uint32_t gH = get_compile_time_arg_val(2);
constexpr uint32_t gK = get_compile_time_arg_val(3);
constexpr uint32_t gV = get_compile_time_arg_val(4);
constexpr uint32_t Ct = get_compile_time_arg_val(5);  // block_chunk_tiles
constexpr uint32_t Kt = get_compile_time_arg_val(6);  // block_key_tiles
constexpr uint32_t Vt = get_compile_time_arg_val(7);
constexpr uint32_t Vb = get_compile_time_arg_val(8);   // block_val_tiles
constexpr uint32_t NVB = get_compile_time_arg_val(9);  // num_v_blocks
constexpr uint32_t NC = get_compile_time_arg_val(10);  // tensor_chunks
constexpr uint32_t BH = get_compile_time_arg_val(11);
constexpr uint32_t NEUMANN_STEPS = get_compile_time_arg_val(12);
constexpr uint32_t ESZ = get_compile_time_arg_val(13);  // input element bytes
constexpr uint32_t ROW_SPAN_STRIDE = get_compile_time_arg_val(14);
constexpr uint32_t GATHER_TOKENS = get_compile_time_arg_val(15);
constexpr uint32_t GATHER_DEPTH = get_compile_time_arg_val(16);
constexpr uint32_t DEST_LIMIT = get_compile_time_arg_val(17);
constexpr uint32_t HAS_H0 = get_compile_time_arg_val(18);
constexpr uint32_t HAS_DHT = get_compile_time_arg_val(19);
constexpr uint32_t BLOCK_CHUNKS = get_compile_time_arg_val(20);
constexpr uint32_t EGRESS_DEPTH = get_compile_time_arg_val(21);
constexpr uint32_t Tt = get_compile_time_arg_val(22);

constexpr uint32_t SC_ATTN = get_compile_time_arg_val(23);
constexpr uint32_t SC_KCD = get_compile_time_arg_val(24);
constexpr uint32_t SC_P = get_compile_time_arg_val(25);
constexpr uint32_t SC_VEC = get_compile_time_arg_val(26);
constexpr uint32_t SC_VCORR = get_compile_time_arg_val(27);
constexpr uint32_t SC_U = get_compile_time_arg_val(28);
constexpr uint32_t SC_C = get_compile_time_arg_val(29);
constexpr uint32_t SC_S = get_compile_time_arg_val(30);
constexpr uint32_t SC_DS = get_compile_time_arg_val(31);
constexpr uint32_t SC_VNEW = get_compile_time_arg_val(32);
constexpr uint32_t SC_DVNEW = get_compile_time_arg_val(33);
constexpr uint32_t SI_Q = get_compile_time_arg_val(34);
constexpr uint32_t SI_K = get_compile_time_arg_val(35);
constexpr uint32_t SI_V = get_compile_time_arg_val(36);
constexpr uint32_t SI_DO = get_compile_time_arg_val(37);
constexpr uint32_t SEM_PREP = get_compile_time_arg_val(38);
constexpr uint32_t SEM_SCAN = get_compile_time_arg_val(39);

// ---------------------------------------------------------------------------
// Uniform CB block sizes and named-slot counts.
//
// These are SOLVED ON HOST (`_cb_blocks()` / `NUM_*_SLOTS` in the program
// descriptor) and arrive here as compile-time args.  The device never
// recomputes a sizing formula: every CB transfers ONE uniform block size, the
// host sizes the CB from exactly these values, and a divergence between the two
// sides would be a fifo wrap (a hang), not a warning -- so there is exactly one
// definition and both sides read it.
// ---------------------------------------------------------------------------
constexpr uint32_t NCONST = get_compile_time_arg_val(40);    // cb_const pages
constexpr uint32_t NCOL = get_compile_time_arg_val(41);      // cb_colones pages
constexpr uint32_t MAXV = get_compile_time_arg_val(42);      // cb_vin block
constexpr uint32_t LVB = get_compile_time_arg_val(43);       // cb_load_vb block
constexpr uint32_t LITEM = get_compile_time_arg_val(44);     // cb_load_item block
constexpr uint32_t MAXBLK = get_compile_time_arg_val(45);    // cb_egr block
constexpr uint32_t MAXBLK_G = get_compile_time_arg_val(46);  // cb_gegr block
constexpr uint32_t NUM_CONST_MASKS = get_compile_time_arg_val(47);
constexpr uint32_t VECA_BLOCKS = get_compile_time_arg_val(48);
constexpr uint32_t VECB_BLOCKS = get_compile_time_arg_val(49);

constexpr uint32_t CT_ACC_BASE = 50;

// derived
constexpr uint32_t CHUNK = Ct * 32;  // tokens per chunk
constexpr uint32_t CtCt = Ct * Ct;
constexpr uint32_t CtKt = Ct * Kt;
constexpr uint32_t CtVb = Ct * Vb;
constexpr uint32_t KtVb = Kt * Vb;
constexpr uint32_t CtVt = Ct * Vt;
constexpr uint32_t KtVt = Kt * Vt;

// per-item scratch strides (must mirror _ScratchMap)
constexpr uint32_t ST_ATTN = CtCt;
constexpr uint32_t ST_KCD = CtKt;
constexpr uint32_t ST_P = CtKt;
constexpr uint32_t ST_VEC = 4 * Ct;  // decay, beta, dc1, rmg
constexpr uint32_t ST_VCORR = CtVt;
constexpr uint32_t ST_U = CtVt;
constexpr uint32_t ST_C = KtVt;
constexpr uint32_t ST_S = KtVt;
constexpr uint32_t ST_DS = KtVt;
constexpr uint32_t ST_VNEW = CtVt;
constexpr uint32_t ST_DVNEW = CtVt;

// ---------------------------------------------------------------------------
// Circular buffers (must mirror the descriptor's CB_* constants)
// ---------------------------------------------------------------------------
constexpr uint32_t cb_const = 0;
constexpr uint32_t cb_colones = 1;
constexpr uint32_t cb_gather = 2;
constexpr uint32_t cb_qin = 3;
constexpr uint32_t cb_kin = 4;
constexpr uint32_t cb_vin = 5;
constexpr uint32_t cb_doin = 6;
constexpr uint32_t cb_gatein = 7;
constexpr uint32_t cb_load_item = 8;
constexpr uint32_t cb_load_vb = 9;
constexpr uint32_t cb_ka = 10;
constexpr uint32_t cb_kb = 11;
constexpr uint32_t cb_kc = 12;
constexpr uint32_t cb_kd = 13;
constexpr uint32_t cb_ke = 14;
constexpr uint32_t cb_kf = 15;
constexpr uint32_t cb_ktr = 16;
constexpr uint32_t cb_va = 17;
constexpr uint32_t cb_vb = 18;
constexpr uint32_t cb_vc = 19;
constexpr uint32_t cb_ca = 20;
constexpr uint32_t cb_cbL = 21;
constexpr uint32_t cb_cc = 22;
constexpr uint32_t cb_cd = 23;
constexpr uint32_t cb_ce = 24;
constexpr uint32_t cb_sa = 25;
constexpr uint32_t cb_sb = 26;
constexpr uint32_t cb_veca = 27;
constexpr uint32_t cb_vecb = 28;
constexpr uint32_t cb_vecc = 29;
constexpr uint32_t cb_egr = 30;
constexpr uint32_t cb_gegr = 31;

// cb_const block offsets (tiles).  NUM_CONST_MASKS [C,C] mask blocks first, then
// the three single-tile / Ct-tile constants; all offsets derive from the
// host-supplied mask count, so adding a mask moves them together.
constexpr uint32_t CST_LT = 0;          // lower-triangular ones, diagonal included
constexpr uint32_t CST_NSTRICT = CtCt;  // -1 on the strictly-lower triangle
constexpr uint32_t CST_UT = 2 * CtCt;   // upper-triangular ones, diagonal included
constexpr uint32_t CST_EYE = 3 * CtCt;
constexpr uint32_t CST_BIAS = 4 * CtCt;  // 0 on tril-inclusive, -1e4 on strict upper
constexpr uint32_t CST_SUT = 5 * CtCt;   // strictly-upper ones (exclusive reverse cumsum)
static_assert(NUM_CONST_MASKS == 6, "cb_const mask layout above assumes six [C,C] mask blocks");
constexpr uint32_t CST_SCALE = NUM_CONST_MASKS * CtCt;        // column 0 == scale (stride-0 COL vector)
constexpr uint32_t CST_ZERO = NUM_CONST_MASKS * CtCt + 1;     // all zeros
constexpr uint32_t CST_ROWONES = NUM_CONST_MASKS * CtCt + 2;  // Ct tiles, row 0 == 1 (outer-product operand)

// cb_veca: the per-item decay column list.  EXACTLY VECA_BLOCKS single-block
// pushes per item, all popped at the end -- same wrap rule as cb_vecb.
constexpr uint32_t VA_DECAY = 0;
constexpr uint32_t VA_GAMMA = Ct;
constexpr uint32_t VA_W = 2 * Ct;
constexpr uint32_t VA_BETA = 3 * Ct;
constexpr uint32_t VA_DC1 = 4 * Ct;  // tile VA_DC1 has [0,0] == log(Gamma)
static_assert(VECA_BLOCKS == 5, "cb_veca slot layout above assumes five column slots");

// cb_vecb slots (stage-G column accumulators), Ct tiles each
// cb_vecb: the stage-G column list.  EXACTLY these ten single-block slots are
// pushed per item and all ten are popped at the end, so the fifo always cycles
// back to its base and no block can straddle the wrap point.
constexpr uint32_t VB_DBETA_V = 0;   // rowsum_V(dv_beta . v), summed over V blocks
constexpr uint32_t VB_S1 = Ct;       // rowsum(ndU . U)
constexpr uint32_t VB_DGG = 2 * Ct;  // d(gamma) * gamma
constexpr uint32_t VB_RMC = 3 * Ct;  // rowsum(R - R^T)
constexpr uint32_t VB_DWW = 4 * Ct;  // d(w) * w
constexpr uint32_t VB_DBETA = 5 * Ct;
constexpr uint32_t VB_GAMMA = 6 * Ct;  // Gamma tile
constexpr uint32_t VB_DGGAM = 7 * Ct;  // LT_ones @ (dGamma * Gamma), as a column
constexpr uint32_t VB_UTE = 8 * Ct;    // UT_ones @ (rowsum(R - R^T) + dgg)
static_assert(VECB_BLOCKS == 9, "cb_vecb slot layout above assumes nine column slots");

// ---------------------------------------------------------------------------
// Tile address math
// ---------------------------------------------------------------------------

// Element offset of (row r, col c) inside a 32x32 tile stored as four 16x16
// faces in (face_r, face_c) row-major order.
FORCE_INLINE uint32_t tile_elem_off(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256u + ((r & 15u) << 4) + (c & 15u);
}

// Byte offset of the start of the first 16-element run of row `r`.  The second
// run sits exactly 256 elements later.
FORCE_INLINE uint32_t row_run0_bytes(uint32_t r) { return tile_elem_off(r, 0) * ESZ; }
constexpr uint32_t ROW_RUN_GAP_BYTES = 256u * ESZ;
constexpr uint32_t ROW_RUN_BYTES = 16u * ESZ;
constexpr uint32_t ROW_SPAN_BYTES = 272u * ESZ;

// Copy one 16-element face-row run between two 4-byte-aligned L1 addresses.
//
// Fully unrolled on purpose.  The reader is RISC-ISSUE bound, not NoC bound
// (measured ~280 ns per gathered row against a ~1 KB read), so the instruction
// count of this copy is a first-order term in the op's dominant stage; at -Os a
// rolled loop costs ~5 instructions per word.
constexpr uint32_t ROW_RUN_WORDS = (16u * ESZ) >> 2;
FORCE_INLINE void copy_row_run(uint32_t dst, uint32_t src) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
#pragma GCC unroll 16
    for (uint32_t i = 0; i < ROW_RUN_WORDS; ++i) {
        d[i] = s[i];
    }
}

// One face-row staging window (the only non-tile-paged buffer in the op).
constexpr uint32_t GATHER_SLOT_BYTES = GATHER_TOKENS * ROW_SPAN_STRIDE + 64;
