// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// chunk_gated_delta_rule_fwd — shared compile-time geometry, CB ids and tile-address math.
//
// Included by all three kernels.  The compile-time argument block (indices 0..CT_ACC_BASE-1)
// is IDENTICAL across the reader, writer and compute binaries: it is built once on the host
// (`_common_ct_args()` in the program descriptor) and every kernel reads it through this header.
// Each dataflow kernel appends its own TensorAccessorArgs after CT_ACC_BASE.
//
// Every block extent / depth / quantum below is SOLVED ON HOST and arrives here; the device never
// recomputes a sizing formula (a divergence would be a fifo wrap — a hang — not a warning).

#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// Compile-time geometry (mirrors `_common_ct_args` in the program descriptor)
// ---------------------------------------------------------------------------
constexpr uint32_t gT = get_compile_time_arg_val(0);           // tokens
constexpr uint32_t gH = get_compile_time_arg_val(1);           // heads
constexpr uint32_t Ct = get_compile_time_arg_val(2);           // block_chunk_tiles  (chunk_size / 32)
constexpr uint32_t Kt = get_compile_time_arg_val(3);           // block_key_tiles
constexpr uint32_t Vt = get_compile_time_arg_val(4);           // value tiles
constexpr uint32_t Vi = get_compile_time_arg_val(5);           // item_block_val_tiles  (P, E)
constexpr uint32_t NVI = get_compile_time_arg_val(6);          // num_item_v_blocks = Vt / Vi
constexpr uint32_t Vs = get_compile_time_arg_val(7);           // scan_block_val_tiles (S)
constexpr uint32_t NV = get_compile_time_arg_val(8);           // scan units per (bh) = Vt / Vs
constexpr uint32_t NC = get_compile_time_arg_val(9);           // chunks
constexpr uint32_t BH = get_compile_time_arg_val(10);          // B * H
constexpr uint32_t NS = get_compile_time_arg_val(11);          // ready segments (handoff granularity)
constexpr uint32_t SEG_CHUNKS = get_compile_time_arg_val(12);  // chunks per segment
constexpr uint32_t NEUMANN_STEPS = get_compile_time_arg_val(13);
constexpr uint32_t ESZ = get_compile_time_arg_val(14);  // input element bytes
constexpr uint32_t ROW_SPAN_STRIDE = get_compile_time_arg_val(15);
constexpr uint32_t GATHER_TOKENS = get_compile_time_arg_val(16);  // gather_stage_tokens
constexpr uint32_t GATHER_DEPTH = get_compile_time_arg_val(17);
constexpr uint32_t DEST_LIMIT = get_compile_time_arg_val(18);
constexpr uint32_t HAS_H0 = get_compile_time_arg_val(19);
constexpr uint32_t Tt = get_compile_time_arg_val(20);       // token tiles of the rank-3 tensors
constexpr uint32_t QV = get_compile_time_arg_val(21);       // cb_vblock_in push quantum
constexpr uint32_t QF = get_compile_time_arg_val(22);       // cb_scratch_egress push quantum
constexpr uint32_t QO = get_compile_time_arg_val(23);       // cb_out_egress push quantum
constexpr uint32_t SC_NKCD = get_compile_time_arg_val(24);  // scratch tile bases
constexpr uint32_t SC_PT = get_compile_time_arg_val(25);
constexpr uint32_t SC_GAM = get_compile_time_arg_val(26);
constexpr uint32_t SC_VCORR = get_compile_time_arg_val(27);
constexpr uint32_t SC_QD = get_compile_time_arg_val(28);
constexpr uint32_t SC_INTRA = get_compile_time_arg_val(29);
constexpr uint32_t SC_VNEW = get_compile_time_arg_val(30);
constexpr uint32_t SEM_READY_BASE = get_compile_time_arg_val(31);  // sem_ready[j] = SEM_READY_BASE + j
constexpr uint32_t SEM_DONE_BASE = get_compile_time_arg_val(32);   // sem_done[j]  = SEM_DONE_BASE + j
constexpr uint32_t BLOCK_CHUNKS = get_compile_time_arg_val(33);
constexpr uint32_t BLOCK_HEADS = get_compile_time_arg_val(34);
constexpr uint32_t NCONST = get_compile_time_arg_val(35);     // cb_const pages
constexpr uint32_t VEC_PAGES = get_compile_time_arg_val(36);  // cb_vec pages (one per-item block)

constexpr uint32_t CT_ACC_BASE = 37;

// Regime R1 is built on single-chunk, single-head items.  The knobs exist (host constants
// BLOCK_CHUNKS / BLOCK_HEADS) and travel here so a later refinement turns them in ONE place; the
// kernels implement exactly the value 1 today and say so loudly instead of silently mis-indexing.
static_assert(BLOCK_CHUNKS == 1, "R1 kernels implement block_chunks == 1 (multi-chunk items: refinement)");
static_assert(BLOCK_HEADS == 1, "R1 kernels implement block_heads == 1 (multi-head blocks are regime R3)");

// derived
constexpr uint32_t CHUNK = Ct * 32;
constexpr uint32_t CtCt = Ct * Ct;
constexpr uint32_t CtKt = Ct * Kt;
constexpr uint32_t CtVi = Ct * Vi;
constexpr uint32_t CtVs = Ct * Vs;
constexpr uint32_t KtVs = Kt * Vs;
constexpr uint32_t KtVi = Kt * Vi;
constexpr uint32_t CtVt = Ct * Vt;
constexpr uint32_t KtVt = Kt * Vt;

constexpr uint32_t F32_TILE = 32 * 32 * 4;
constexpr uint32_t IN_TILE = 32 * 32 * ESZ;

// ---------------------------------------------------------------------------
// Circular buffers (must mirror the descriptor's CB_* constants)
// ---------------------------------------------------------------------------
constexpr uint32_t cb_const = 0;
constexpr uint32_t cb_gather_stage = 1;
constexpr uint32_t cb_scalar_stage = 2;
constexpr uint32_t cb_q_in = 3;
constexpr uint32_t cb_k_in = 4;
constexpr uint32_t cb_vblock_in = 5;
constexpr uint32_t cb_gate_in = 6;
constexpr uint32_t cb_vec = 7;
constexpr uint32_t cb_qs = 8;
constexpr uint32_t cb_kb = 9;
constexpr uint32_t cb_kw = 10;
constexpr uint32_t cb_L = 11;
constexpr uint32_t cb_cc_a = 12;
constexpr uint32_t cb_cc_b = 13;
constexpr uint32_t cb_T = 14;
constexpr uint32_t cb_pow = 15;
constexpr uint32_t cb_vmat = 16;
constexpr uint32_t cb_intra_in = 17;
constexpr uint32_t cb_vnew_in = 18;
constexpr uint32_t cb_kmat_in = 20;
constexpr uint32_t cb_scan_pt = 21;
constexpr uint32_t cb_scan_vcorr = 22;
constexpr uint32_t cb_scan_gamma = 23;
constexpr uint32_t cb_state = 25;
constexpr uint32_t cb_scan_vnew = 26;
constexpr uint32_t cb_scratch_egress = 27;
constexpr uint32_t cb_out_egress = 28;

// cb_const block offsets (tiles): four [C,C] masks, a [1,C] row of all-ones tiles, and one tile
// whose row 0 is ones (the column -> full-width replication operand).
constexpr uint32_t CST_EYE = 0;
constexpr uint32_t CST_LT = CtCt;              // inclusive lower-triangular ones
constexpr uint32_t CST_SL = 2 * CtCt;          // strict lower-triangular ones
constexpr uint32_t CST_SU = 3 * CtCt;          // strict upper-triangular ones
constexpr uint32_t CST_ONES = 4 * CtCt;        // Ct all-ones tiles
constexpr uint32_t CST_EROW0 = 4 * CtCt + Ct;  // row 0 == 1, rest 0
static_assert(NCONST == 4 * CtCt + Ct + 1, "cb_const layout above must match the host page count");

// cb_vec: the per-item gate-column block (one push of VEC_PAGES per item).
constexpr uint32_t V_GAMMA = 0;       // Ct tiles, exp(decay)        (column 0 valid)
constexpr uint32_t V_W = Ct;          // Ct tiles, exp(sum_{u>t} g)  (column 0 valid)
constexpr uint32_t V_GFULL = 2 * Ct;  // 1 tile,   exp(sum g) in EVERY element
static_assert(VEC_PAGES == 2 * Ct + 1, "cb_vec layout above must match the host page count");

// ---------------------------------------------------------------------------
// Tile address math
// ---------------------------------------------------------------------------

// Element offset of (row r, col c) inside a 32x32 tile stored as four 16x16 faces.
FORCE_INLINE uint32_t tile_elem_off(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256u + ((r & 15u) << 4) + (c & 15u);
}

// Byte offset of the first 16-element run of tile row `r`; the second run is 256 elements later.
FORCE_INLINE uint32_t row_run0_bytes(uint32_t r) { return tile_elem_off(r, 0) * ESZ; }
constexpr uint32_t ROW_RUN_GAP_BYTES = 256u * ESZ;
constexpr uint32_t ROW_RUN_BYTES = 16u * ESZ;
constexpr uint32_t ROW_SPAN_BYTES = 272u * ESZ;
constexpr uint32_t ROW_RUN_WORDS = (16u * ESZ) >> 2;

// One face-row staging window (the only non-tile-paged buffer of the reader).
constexpr uint32_t GATHER_SLOT_BYTES = GATHER_TOKENS * ROW_SPAN_STRIDE + 64;

// Segment of chunk i (the P->S / S->E handoff granularity).
FORCE_INLINE uint32_t seg_of(uint32_t i) { return i / SEG_CHUNKS; }
FORCE_INLINE uint32_t seg_begin(uint32_t j) { return j * SEG_CHUNKS; }
FORCE_INLINE uint32_t seg_end(uint32_t j) {
    const uint32_t e = (j + 1) * SEG_CHUNKS;
    return e < NC ? e : NC;
}
