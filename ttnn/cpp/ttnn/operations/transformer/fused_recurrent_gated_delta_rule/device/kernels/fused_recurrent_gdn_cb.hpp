// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Circular-buffer plan of the GDN decode op (fused_recurrent_gated_delta_rule), shared by the program factory and the
// reader / compute / writer kernels (and by the LLK gtests of the compute passes). Host and device compile this header,
// so it holds constants only.
//
// All CBs are fp32. The token vectors travel in tiny tiles: a 16x32 tile is face 0 (rows 0-15, cols 0-15, 1 KB) then
// face 1 (cols 16-31); row r of face f sits at byte f*1024 + r*64. A 32x16 tile is face 0 (rows 0-15) then face 1
// (rows 16-31); element (r, 0) sits at byte (r/16)*1024 + (r%16)*64. A 32x32 tile is faces 0..3 (rows 0-15 cols 0-15,
// rows 0-15 cols 16-31, rows 16-31 cols 0-15, rows 16-31 cols 16-31); element (r, 0) sits at byte
// (r/16)*2048 + (r%16)*64.
//
// Per token (Kt = K/32, Vt = V/32, kv = Kt*Vt):
// clang-format off
//   id    name       tile   count     contents                                        producer -> consumer
//   c_0   qk16       16x32  Kt x 2    row 0 = k_hat[ki*32..+32], row 2 = q_tilde[..]  reader -> pass 1 in0, constants in0
//   c_1   kT16       32x16  Kt x 2    column 0 = k_hat[ki*32..+32]                    reader -> pass 2 in0
//   c_2   qcol32     32x32  Kt x 2    column 0 = q_tilde[..]                          reader -> constants pass in1
//   c_3   v16        16x32  Vt x 2    row 0 = v[j*32..+32]                            reader -> vector pass (utd)
//   c_4   gcol32     32x32  1 x 2     column 0 (all 32 rows) = g                      reader -> constants pass (utd)
//   c_5   bcol16     16x32  1 x 2     column 0 (rows 0-15) = beta                     reader -> vector pass (utd)
//   c_6   S          32x32  kv        the state at t = 0                              reader -> pass 2 seed (utd); c_20 is its matmul view
//   c_7   ones32     32x32  1         all 1.0, written once at kernel start           reader -> constants pass in0
//   c_8   zero32     32x32  1         zeros, written once                             reader -> compute (utd)
//   c_9   ecol32     32x32  1         exp(gcol32): column 0 = e, elsewhere 1          compute -> vector pass, pass 2 (utd)
//   c_10  (freed)    -      -         was r16; the fused read+vector pass keeps r in DEST, so no CB is allocated
//   c_11  craw16     16x32  1         [0, 0] = c = q_tilde . k_hat                    compute -> constants pass
//   c_12  (freed)    -      -         was ccol32; c goes from DEST straight into ebc16, so no CB is allocated
//   c_13  u16        16x32  Vt        row 0 = u, rows 1-15 exactly zero               compute -> pass 2 in1
//   c_14  s2         32x32  kv        state ring (ping)                               compute -> compute (utd); c_21 is its matmul view
//   c_15  s3         32x32  kv        state ring (pong)                               compute -> compute (utd); c_22 is its matmul view
//   c_16  out16      16x32  Vt x 2    row 0 = o                                       compute -> writer (2048 B per V tile)
//   c_17  state_out  32x32  kv x 2    per-token or final state                        compute -> writer
//   c_18  blkidx     n/a    1 page    ring-mode block-index scratch (uint32 [BH])     reader
//   c_19  ebc16      16x32  1         column 0: rows 0-3 = e, 4-7 = beta, 8-11 = c    compute constants pass -> vector pass (utd)
//   c_20  S_mm       32x32  kv        matmul view of c_6 (same L1, never pushed)      pass 1 in1
//   c_21  s2_mm      32x32  kv        matmul view of c_14 (same L1, never pushed)     pass 1 in1
//   c_22  s3_mm      32x32  kv        matmul view of c_15 (same L1, never pushed)     pass 1 in1
// clang-format on
// "utd": the factory lists the CB with UnpackToDestMode::UnpackToDestFp32 (it is only ever read by copy_tile into
// DEST). Every CB sized exactly `count` tiles cycles through the same L1 slots token after token, so the reader writes
// the zero paddings once at kernel start and only the live rows / columns per token.
//
// State CBs (S, s2, s3) are each one allocation under two ids (one CircularBufferConfig, two buffer indices), a
// StateCb below. The `utd` id is UnpackToDestFp32 and owns all the flow control: it is reserved, pushed, waited on and
// popped. The `mm` id is a format view for the read-pass matmul (a utd CB cannot be a matmul operand: the jit-build
// gives it the Float32 unpack format and the matmul reads garbage from SrcA; measured in test_gdn_decode_math.cpp)
// and is never reserved, pushed, waited on or popped. That is sound because a state CB holds exactly one state (kv
// tiles, single-buffered) and every token produces and consumes it whole: the read and write pointers of both ids
// start at the CB base and wrap back to it after kv tiles, so tile i sits at base + i tiles for every id, every token.
// The unpack thread waits on the utd id before it unpacks through the mm view, and that wait is the only ordering the
// view needs. The factory checks the one-state sizing (add_state_cb); the compute passes check the pointers under LLK
// asserts (gdn_decode::check_state_views). Double-buffering a state CB would break the invariant; the view would then
// have to copy the utd id's read pointer before each read pass instead (the matmul kernels' evil_set_read_ptr form).
#pragma once

#include <array>
#include <cstdint>

namespace gdn_decode_cb {

constexpr uint32_t qk16 = 0;
constexpr uint32_t kT16 = 1;
constexpr uint32_t qcol32 = 2;
constexpr uint32_t v16 = 3;
constexpr uint32_t gcol32 = 4;
constexpr uint32_t bcol16 = 5;
constexpr uint32_t S = 6;
constexpr uint32_t ones32 = 7;
constexpr uint32_t zero32 = 8;
constexpr uint32_t ecol32 = 9;
constexpr uint32_t freed_r16 = 10;  // not allocated; the math header's TokenCbs still names the field
constexpr uint32_t craw16 = 11;
constexpr uint32_t freed_ccol32 = 12;  // not allocated; the math header's TokenCbs still names the field
constexpr uint32_t u16 = 13;
constexpr uint32_t s2 = 14;
constexpr uint32_t s3 = 15;
constexpr uint32_t out16 = 16;
constexpr uint32_t state_out = 17;
constexpr uint32_t blkidx = 18;
constexpr uint32_t ebc16 = 19;  // the vector pass's constants tile (e, beta, c in column 0), compute-internal
constexpr uint32_t S_mm = 20;   // matmul-operand id of S (same L1 as c_6)
constexpr uint32_t s2_mm = 21;  // matmul-operand id of s2 (same L1 as c_14)
constexpr uint32_t s3_mm = 22;  // matmul-operand id of s3 (same L1 as c_15)

constexpr uint32_t num_cbs = 23;  // c_0 .. c_22 (blkidx is only allocated in ring mode)
// A state CB under its two ids (see the note above). single(id) is the degenerate pair for a CB that is only ever
// copied, never a matmul operand (a test's output CB).
struct StateCb {
    uint32_t utd;
    uint32_t mm;
    static constexpr StateCb single(uint32_t id) { return StateCb{id, id}; }
};
constexpr StateCb state_S{S, S_mm};
constexpr StateCb state_s2{s2, s2_mm};
constexpr StateCb state_s3{s3, s3_mm};
constexpr std::array<StateCb, 3> state_cbs = {{state_S, state_s2, state_s3}};
// The compute kernel's state ring: the reader's S at t = 0, then s2 / s3 alternate (a token reads one, writes the
// other).
constexpr StateCb state_cur(uint32_t t) { return t == 0 ? state_S : ((t & 1u) ? state_s2 : state_s3); }
constexpr StateCb state_next(uint32_t t) { return (t & 1u) ? state_s3 : state_s2; }

// The reader's per-token CBs (qk16, kT16, qcol32, v16, gcol32, bcol16) hold this many tokens, so the reader builds
// token t+1 while the compute consumes token t (decode_token pops its inputs only at its end). The reader zero-fills
// every slot set once at kernel start; the per-token reserve/push walks the slot sets in a fixed order.
constexpr uint32_t token_cb_nbuf = 2;

// Tile dims {rows, cols} per CB id (blkidx is not a tile CB; its entry is a placeholder).
constexpr std::array<uint32_t, 2> tile_16x32 = {16, 32};
constexpr std::array<uint32_t, 2> tile_32x16 = {32, 16};
constexpr std::array<uint32_t, 2> tile_32x32 = {32, 32};

constexpr std::array<std::array<uint32_t, 2>, num_cbs> tile_dims = {{
    tile_16x32,  // c_0  qk16
    tile_32x16,  // c_1  kT16
    tile_32x32,  // c_2  qcol32
    tile_16x32,  // c_3  v16
    tile_32x32,  // c_4  gcol32
    tile_16x32,  // c_5  bcol16
    tile_32x32,  // c_6  S
    tile_32x32,  // c_7  ones32
    tile_32x32,  // c_8  zero32
    tile_32x32,  // c_9  ecol32
    tile_16x32,  // c_10 (freed, was r16)
    tile_16x32,  // c_11 craw16
    tile_32x32,  // c_12 (freed, was ccol32)
    tile_16x32,  // c_13 u16
    tile_32x32,  // c_14 s2
    tile_32x32,  // c_15 s3
    tile_16x32,  // c_16 out16
    tile_32x32,  // c_17 state_out
    tile_32x32,  // c_18 blkidx (placeholder)
    tile_16x32,  // c_19 ebc16
    tile_32x32,  // c_20 S_mm
    tile_32x32,  // c_21 s2_mm
    tile_32x32,  // c_22 s3_mm
}};

// CBs read only by copy_tile into DEST: the factory lists them with UnpackToDestMode::UnpackToDestFp32.
constexpr std::array<bool, num_cbs> unpack_to_dest_fp32 = {{
    false,  // c_0  qk16      (matmul in0)
    false,  // c_1  kT16      (matmul in0)
    false,  // c_2  qcol32    (matmul in1)
    true,   // c_3  v16
    true,   // c_4  gcol32
    true,   // c_5  bcol16
    true,   // c_6  S
    false,  // c_7  ones32    (bcast-scalar in0)
    true,   // c_8  zero32
    true,   // c_9  ecol32
    false,  // c_10 (freed, was r16)
    false,  // c_11 craw16    (bcast-scalar in1)
    false,  // c_12 (freed, was ccol32)
    false,  // c_13 u16       (matmul in1)
    true,   // c_14 s2
    true,   // c_15 s3
    false,  // c_16 out16
    false,  // c_17 state_out
    false,  // c_18 blkidx
    true,   // c_19 ebc16
    false,  // c_20 S_mm      (matmul view of c_6)
    false,  // c_21 s2_mm     (matmul view of c_14)
    false,  // c_22 s3_mm     (matmul view of c_15)
}};
// Every state CB: the utd id unpacks to DEST, its matmul view does not, and the two share the tile dims.
constexpr bool state_views_consistent() {
    for (const StateCb& s : state_cbs) {
        // Element-wise: the kernels compile as C++17, where std::array's operator!= is not constexpr.
        const bool same_dims = tile_dims[s.utd][0] == tile_dims[s.mm][0] && tile_dims[s.utd][1] == tile_dims[s.mm][1];
        if (!unpack_to_dest_fp32[s.utd] || unpack_to_dest_fp32[s.mm] || !same_dims) {
            return false;
        }
    }
    return true;
}
static_assert(state_views_consistent(), "a state CB's utd id must be UnpackToDestFp32 and its mm view must not be");

// Byte size of an fp32 tile of the given dims.
constexpr uint32_t tile_bytes(std::array<uint32_t, 2> dims) { return dims[0] * dims[1] * 4u; }

// Byte offsets inside the tiles the reader builds (see the layout notes above).
constexpr uint32_t face_bytes_16x16 = 16u * 16u * 4u;  // 1024
constexpr uint32_t row_bytes = 16u * 4u;               // 64: one 16-element face row

// 16x32 tile: byte offset of row r, columns [16*f, 16*f + 16).
constexpr uint32_t off_16x32(uint32_t r, uint32_t f) { return f * face_bytes_16x16 + r * row_bytes; }
// 32x16 tile: byte offset of element (r, 0).
constexpr uint32_t off_32x16_col0(uint32_t r) { return (r / 16u) * face_bytes_16x16 + (r % 16u) * row_bytes; }
// 32x32 tile: byte offset of element (r, 0).
constexpr uint32_t off_32x32_col0(uint32_t r) { return (r / 16u) * 2u * face_bytes_16x16 + (r % 16u) * row_bytes; }
// 32x32 tile: byte offset of row r, columns [16*f, 16*f + 16) for f in {0, 1} (r < 16) -- the live row 0 of today's
// host-prepared [1, D] tiles sits at off_32x32_row(0, 0) and off_32x32_row(0, 1).
constexpr uint32_t off_32x32_row(uint32_t r, uint32_t f) {
    return (r / 16u) * 2u * face_bytes_16x16 + f * face_bytes_16x16 + (r % 16u) * row_bytes;
}

}  // namespace gdn_decode_cb
