// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader: initial state S [K,V] once (from s0, the ring, or zeros built in-kernel), then per token the tiny-tile
// operands of the two-pass recurrence (see ../fused_recurrent_gdn_cb.hpp for the plan). All CB tiles are fp32.
// Device 2.0 API.
//
// Source layout: the layer's own tensors, TILE, interleaved, 32x32 tiles of bf16 or fp32.
//   q, k [B, T, H, K]: head kh's row of token (b, t) is row kh % 32 of tile row kh / 32; the page of its ki-th K
//        tile is ((b*T + t) * ceil(H/32) + kh/32) * Kt + ki. Core (b, hv) reads key head kh = hv / (HV / H) (GQA).
//   v    [B, T, HV, V]: the same with hv, HV and Vt.
//   g, beta [B, T, HV]: tiles over (T, HV); the element of (t, hv) sits at row t % 32, column hv % 32 of page
//        (b * ceil(T/32) + t/32) * ceil(HV/32) + hv/32.
// Only the live rows are read: a face row is 16 elements, 64 B of fp32 or 32 B of bf16; DRAM reads must be 64-byte
// aligned, so fp32 rows go straight into the CB tiles and bf16 rows arrive as their 64-byte chunk in the raw64
// scratch and are widened (bf16 bits << 16) into the fp32 tiles by the RISC. The column tiles (kT16, qcol32, gcol32,
// bcol16) are built by a RISC loop from the landed rows. The zero paddings of every token CB are written once at
// kernel start: the CBs hold token_cb_nbuf slot sets that the per-token reserve/push walks in a fixed order, so a
// padding is never overwritten (at T = 1 only the first slot set is used and zeroed).
//
// Per token this is 4*Kt + 2*Vt + 2 NoC reads of 64 bytes (26 at Kt = Vt = 4) under one barrier; token 0's reads are
// issued together with the state's so their latency overlaps.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/debug/assert.h"
#include "api/tensor/noc_traits.h"

#include "../fused_recurrent_gdn_cb.hpp"

namespace cb = gdn_decode_cb;

namespace {

constexpr uint32_t one_f32 = 0x3f800000u;

// The compiler must not move the RISC's L1 accesses across a NoC barrier or a CB push (both are MMIO).
inline void compiler_fence() { asm volatile("" ::: "memory"); }

// bf16 -> fp32: the bf16 bits are the high half of the fp32 word. Element 2i of a row is the low half of word i,
// element 2i + 1 the high half.
inline uint32_t widen(uint32_t word, uint32_t odd) { return odd ? (word & 0xFFFF0000u) : (word << 16); }

// Byte offset of face column f of row r in a 32x32 tile of e-byte elements (faces 0..3 are rows 0-15 / 16-31 by
// columns 0-15 / 16-31; a face is 16 x 16 elements).
constexpr uint32_t tile_row_off(uint32_t r, uint32_t f, uint32_t e) {
    return ((r / 16u) * 2u + f) * 256u * e + (r % 16u) * 16u * e;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t use_blk_idx = get_compile_time_arg_val(2);
    constexpr uint32_t has_state = get_compile_time_arg_val(3);  // 0: S is zeros, built here
    constexpr uint32_t in_bf16 = get_compile_time_arg_val(4);    // q, k, v dtype
    constexpr uint32_t g_bf16 = get_compile_time_arg_val(5);
    constexpr uint32_t b_bf16 = get_compile_time_arg_val(6);
    constexpr uint32_t H = get_compile_time_arg_val(7);
    constexpr uint32_t HV = get_compile_time_arg_val(8);
    constexpr uint32_t G = get_compile_time_arg_val(9);  // HV / H

    constexpr auto q_a = TensorAccessorArgs<10>();
    constexpr auto k_a = TensorAccessorArgs<q_a.next_compile_time_args_offset()>();
    constexpr auto v_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    constexpr auto g_a = TensorAccessorArgs<v_a.next_compile_time_args_offset()>();
    constexpr auto b_a = TensorAccessorArgs<g_a.next_compile_time_args_offset()>();
    constexpr auto s0_a = TensorAccessorArgs<b_a.next_compile_time_args_offset()>();
    constexpr auto idx_a = TensorAccessorArgs<s0_a.next_compile_time_args_offset()>();

    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t T = get_arg_val<uint32_t>(1);
    const uint32_t q_addr = get_arg_val<uint32_t>(2);
    const uint32_t k_addr = get_arg_val<uint32_t>(3);
    const uint32_t v_addr = get_arg_val<uint32_t>(4);
    const uint32_t g_addr = get_arg_val<uint32_t>(5);
    const uint32_t b_addr = get_arg_val<uint32_t>(6);
    const uint32_t s0_addr = get_arg_val<uint32_t>(7);
    // rt 8 = block-index buffer address (0 when absent), rt 9 = its aligned page size, rt 10 = BH.

    // Source tiles are 32x32 of the tensor's dtype; the state's are fp32. The CB tiles are the tiny ones of the plan.
    constexpr uint32_t in_e = in_bf16 ? 2u : 4u;
    constexpr uint32_t g_e = g_bf16 ? 2u : 4u;
    constexpr uint32_t b_e = b_bf16 ? 2u : 4u;
    constexpr uint32_t in_tb = 1024u * in_e;
    constexpr uint32_t s_tb = cb::tile_bytes(cb::tile_32x32);
    const auto q_acc = TensorAccessor(q_a, q_addr, in_tb);
    const auto k_acc = TensorAccessor(k_a, k_addr, in_tb);
    const auto v_acc = TensorAccessor(v_a, v_addr, in_tb);
    const auto g_acc = TensorAccessor(g_a, g_addr, 1024u * g_e);
    const auto b_acc = TensorAccessor(b_a, b_addr, 1024u * b_e);
    const auto s0_acc = TensorAccessor(s0_a, s0_addr, s_tb);

    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t tb16 = cb::tile_bytes(cb::tile_16x32);  // 2048: also the 32x16 tile
    constexpr uint32_t tb32 = cb::tile_bytes(cb::tile_32x32);  // 4096
    constexpr uint32_t chunk = 64;                             // one DRAM-aligned read: a 16-element fp32 face row

    // The head's coordinates in the layer tensors: core h is (b, hv); it reads key head kh.
    const uint32_t b = h / HV;
    const uint32_t hv = h % HV;
    const uint32_t kh = hv / G;
    constexpr uint32_t H_t = (H + 31u) / 32u;
    constexpr uint32_t HV_t = (HV + 31u) / 32u;
    const uint32_t T_t = (T + 31u) / 32u;
    const uint32_t kr = kh % 32u, kpr = kh / 32u;  // row within the tile, tile row
    const uint32_t vr = hv % 32u, vpr = hv / 32u;
    const uint32_t gr = hv % 32u, gpr = hv / 32u;  // g, beta: the column of (t, hv)

    Noc noc;
    CircularBuffer cb_qk(cb::qk16), cb_kT(cb::kT16), cb_qc(cb::qcol32), cb_v(cb::v16);
    CircularBuffer cb_g(cb::gcol32), cb_b(cb::bcol16), cb_ones(cb::ones32), cb_zero(cb::zero32);
    CircularBuffer cb_S(cb::S);  // its matmul view (c_20) is never pushed: see the CB header
    // raw64: the bf16 landing zone (64-byte chunks), reader-private; slot s of a token is at s * chunk.
    constexpr bool any_bf16 = in_bf16 || g_bf16 || b_bf16;
    CircularBuffer cb_raw(cb::raw64);
    constexpr uint32_t raw_k = 0, raw_q = 2 * Kt, raw_v = 4 * Kt, raw_g = 4 * Kt + 2 * Vt, raw_b = raw_g + 1;
    uint32_t raw_base = 0;
    if constexpr (any_bf16) {
        cb_raw.reserve_back(1);
        raw_base = cb_raw.get_write_ptr();
    }

    // ---- once at kernel start: constants and the zero paddings of every per-token CB ----------------------------
    // Reserving the whole CB (nothing is pushed yet) gives the base of all its slot sets; the per-token reserve/push
    // below walks them in a fixed order. zero32 is zeroed by the NoC's zero write; every other padding is a local NoC
    // copy of zero32 in 4 KB pieces (the zero write itself goes in small loopback pieces: 96 KB of it cost 3.7 us).
    cb_zero.reserve_back(1);
    noc.async_write_zeros(cb_zero, tb32);
    noc.write_zeros_l1_barrier();
    const uint64_t zero_src = get_noc_addr(cb_zero.get_write_ptr());
    {
        constexpr uint32_t nb = cb::token_cb_nbuf;
        const uint32_t nb_used = T > 1 ? nb : 1u;  // slot sets the token loop will touch
        auto zero_fill = [&](CircularBuffer& c, uint32_t capacity_tiles, uint32_t bytes) {
            c.reserve_back(capacity_tiles);
            const uint32_t dst = c.get_write_ptr();
            for (uint32_t off = 0; off < bytes; off += tb32) {
                noc_async_read(zero_src, dst + off, (bytes - off < tb32) ? (bytes - off) : tb32);
            }
        };
        zero_fill(cb_qk, nb * Kt, nb_used * Kt * tb16);
        zero_fill(cb_kT, nb * Kt, nb_used * Kt * tb16);
        zero_fill(cb_qc, nb * Kt, nb_used * Kt * tb32);
        zero_fill(cb_v, nb * Vt, nb_used * Vt * tb16);
        zero_fill(cb_g, nb, nb_used * tb32);
        zero_fill(cb_b, nb, nb_used * tb16);
        // ones32: the RISC writes the first 8 rows (512 B), the NoC replicates them into the other 7 x 512 B.
        cb_ones.reserve_back(1);
        const uint32_t ones_base = cb_ones.get_write_ptr();
        {
            tt_l1_ptr uint32_t* p = reinterpret_cast<tt_l1_ptr uint32_t*>(ones_base);
            for (uint32_t i = 0; i < 512 / 4; i++) {
                p[i] = one_f32;
            }
            compiler_fence();
            for (uint32_t off = 512; off < tb32; off += 512) {
                noc_async_read(get_noc_addr(ones_base), ones_base + off, 512);
            }
        }
        noc_async_read_barrier();
        cb_zero.push_back(1);
        cb_ones.push_back(1);
    }

    // ---- per-token tiles: the NoC reads (issue) and the column build + push (finish) ----------------------------
    // One face row of a source tile into the CB tile (fp32: the aligned 64 B row itself) or into a raw64 slot (bf16:
    // the aligned 64-byte chunk that holds the row and its neighbour).
    auto read_row = [&](const auto& acc,
                        uint32_t page,
                        uint32_t r,
                        uint32_t f,
                        CircularBuffer& dst_cb,
                        uint32_t dst_off,
                        uint32_t raw_slot) {
        if constexpr (in_bf16) {
            noc.async_read(
                acc,
                cb_raw,
                chunk,
                {.page_id = page, .offset_bytes = tile_row_off(r, f, 2) & ~(chunk - 1)},
                {.offset_bytes = raw_slot * chunk});
        } else {
            noc.async_read(
                acc,
                dst_cb,
                chunk,
                {.page_id = page, .offset_bytes = tile_row_off(r, f, 4)},
                {.offset_bytes = dst_off});
        }
    };
    auto issue_token_reads = [&](uint32_t t) {
        const uint32_t bt = b * T + t;
        const uint32_t qk_page0 = (bt * H_t + kpr) * Kt;
        const uint32_t v_page0 = (bt * HV_t + vpr) * Vt;
        cb_qk.reserve_back(Kt);
        cb_kT.reserve_back(Kt);
        cb_qc.reserve_back(Kt);
        cb_v.reserve_back(Vt);
        cb_g.reserve_back(1);
        cb_b.reserve_back(1);
        // qk16[ki]: row 0 = k (faces 0 and 1), row 2 = q.
        for (uint32_t ki = 0; ki < Kt; ki++) {
            const uint32_t page = qk_page0 + ki;
            const uint32_t dst = ki * tb16;
            read_row(k_acc, page, kr, 0, cb_qk, dst + cb::off_16x32(0, 0), raw_k + ki * 2);
            read_row(k_acc, page, kr, 1, cb_qk, dst + cb::off_16x32(0, 1), raw_k + ki * 2 + 1);
            read_row(q_acc, page, kr, 0, cb_qk, dst + cb::off_16x32(2, 0), raw_q + ki * 2);
            read_row(q_acc, page, kr, 1, cb_qk, dst + cb::off_16x32(2, 1), raw_q + ki * 2 + 1);
        }
        // v16[j]: row 0 = v.
        for (uint32_t j = 0; j < Vt; j++) {
            const uint32_t page = v_page0 + j;
            const uint32_t dst = j * tb16;
            read_row(v_acc, page, vr, 0, cb_v, dst + cb::off_16x32(0, 0), raw_v + j * 2);
            read_row(v_acc, page, vr, 1, cb_v, dst + cb::off_16x32(0, 1), raw_v + j * 2 + 1);
        }
        // g, beta: the 64-byte chunk holding element (t % 32, hv % 32) of the (b, t, hv) tile. fp32: the face row,
        // landed as row 0 of the column tile (finish_token picks the element and rebuilds the row); bf16: a raw slot.
        const uint32_t gb_page = (b * T_t + t / 32u) * HV_t + gpr;
        const uint32_t tr = t % 32u;
        if constexpr (g_bf16) {
            noc.async_read(
                g_acc,
                cb_raw,
                chunk,
                {.page_id = gb_page, .offset_bytes = tile_row_off(tr, gr / 16u, 2) & ~(chunk - 1)},
                {.offset_bytes = raw_g * chunk});
        } else {
            noc.async_read(
                g_acc,
                cb_g,
                chunk,
                {.page_id = gb_page, .offset_bytes = tile_row_off(tr, gr / 16u, 4)},
                {.offset_bytes = 0});
        }
        if constexpr (b_bf16) {
            noc.async_read(
                b_acc,
                cb_raw,
                chunk,
                {.page_id = gb_page, .offset_bytes = tile_row_off(tr, gr / 16u, 2) & ~(chunk - 1)},
                {.offset_bytes = raw_b * chunk});
        } else {
            noc.async_read(
                b_acc,
                cb_b,
                chunk,
                {.page_id = gb_page, .offset_bytes = tile_row_off(tr, gr / 16u, 4)},
                {.offset_bytes = 0});
        }
    };

    // Widen one landed bf16 row (16 elements at the given half of its 64-byte chunk) into 16 fp32 words.
    auto widen_row = [&](uint32_t raw_slot, uint32_t r, tt_l1_ptr uint32_t* __restrict__ dst) {
        const tt_l1_ptr uint32_t* __restrict__ src =
            reinterpret_cast<const tt_l1_ptr uint32_t*>(raw_base + raw_slot * chunk + ((r % 16u) & 1u) * 32u);
#pragma GCC unroll 8
        for (uint32_t w = 0; w < 8; w++) {
            const uint32_t word = src[w];
            dst[2 * w] = widen(word, 0);
            dst[2 * w + 1] = widen(word, 1);
        }
    };
    // The element (row tr, column gc) of a landed chunk: fp32 chunks hold the face row (16 elements), bf16 chunks two
    // face rows (32 elements).
    auto chunk_element =
        [&](uint32_t raw_slot, const tt_l1_ptr uint32_t* f32_row, uint32_t tr, uint32_t gc, bool bf16) -> uint32_t {
        if (!bf16) {
            return f32_row[gc % 16u];
        }
        const tt_l1_ptr uint32_t* raw = reinterpret_cast<const tt_l1_ptr uint32_t*>(raw_base + raw_slot * chunk);
        const uint32_t idx = ((tr % 16u) & 1u) * 16u + (gc % 16u);
        return widen(raw[idx / 2u], idx & 1u);
    };

    // Call after the reads of issue_token_reads have landed (a NoC read barrier).
    auto finish_token = [&](uint32_t t) {
        compiler_fence();
        if constexpr (in_bf16) {
            for (uint32_t ki = 0; ki < Kt; ki++) {
                tt_l1_ptr uint32_t* qk = reinterpret_cast<tt_l1_ptr uint32_t*>(cb_qk.get_write_ptr() + ki * tb16);
                widen_row(raw_k + ki * 2, kr, qk + cb::off_16x32(0, 0) / 4);
                widen_row(raw_k + ki * 2 + 1, kr, qk + cb::off_16x32(0, 1) / 4);
                widen_row(raw_q + ki * 2, kr, qk + cb::off_16x32(2, 0) / 4);
                widen_row(raw_q + ki * 2 + 1, kr, qk + cb::off_16x32(2, 1) / 4);
            }
            for (uint32_t j = 0; j < Vt; j++) {
                tt_l1_ptr uint32_t* v = reinterpret_cast<tt_l1_ptr uint32_t*>(cb_v.get_write_ptr() + j * tb16);
                widen_row(raw_v + j * 2, vr, v + cb::off_16x32(0, 0) / 4);
                widen_row(raw_v + j * 2 + 1, vr, v + cb::off_16x32(0, 1) / 4);
            }
            compiler_fence();
        }
        // Column tiles from the landed rows: kT16[ki] column 0 = k, qcol32[ki] column 0 = q. Element c of face f of
        // the row tile is column 16f + c, i.e. row 16f + c of the column tiles.
        for (uint32_t ki = 0; ki < Kt; ki++) {
            // __restrict__: the three tiles never overlap, so the compiler may issue a face row's 16 loads before its
            // 16 scattered stores instead of a load-use stall per element.
            const tt_l1_ptr uint32_t* __restrict__ qk =
                reinterpret_cast<const tt_l1_ptr uint32_t*>(cb_qk.get_write_ptr() + ki * tb16);
            tt_l1_ptr uint32_t* __restrict__ kT =
                reinterpret_cast<tt_l1_ptr uint32_t*>(cb_kT.get_write_ptr() + ki * tb16);
            tt_l1_ptr uint32_t* __restrict__ qc =
                reinterpret_cast<tt_l1_ptr uint32_t*>(cb_qc.get_write_ptr() + ki * tb32);
#pragma GCC unroll 16
            for (uint32_t c = 0; c < 16; c++) {
                kT[cb::off_32x16_col0(c) / 4] = qk[cb::off_16x32(0, 0) / 4 + c];
            }
#pragma GCC unroll 16
            for (uint32_t c = 0; c < 16; c++) {
                kT[cb::off_32x16_col0(16 + c) / 4] = qk[cb::off_16x32(0, 1) / 4 + c];
            }
#pragma GCC unroll 16
            for (uint32_t c = 0; c < 16; c++) {
                qc[cb::off_32x32_col0(c) / 4] = qk[cb::off_16x32(2, 0) / 4 + c];
            }
#pragma GCC unroll 16
            for (uint32_t c = 0; c < 16; c++) {
                qc[cb::off_32x32_col0(16 + c) / 4] = qk[cb::off_16x32(2, 1) / 4 + c];
            }
        }
        {
            // gcol32: column 0 (all 32 rows) = g; bcol16: column 0 (rows 0-15) = beta. The landed fp32 face row sits
            // in row 0 of the column tile, so after picking the element the row's other 15 words are zeroed again.
            const uint32_t tr = t % 32u;
            tt_l1_ptr uint32_t* g = reinterpret_cast<tt_l1_ptr uint32_t*>(cb_g.get_write_ptr());
            const uint32_t g_val = chunk_element(raw_g, g, tr, gr, g_bf16 != 0);
#pragma GCC unroll 32
            for (uint32_t r = 0; r < 32; r++) {
                g[cb::off_32x32_col0(r) / 4] = g_val;
            }
#pragma GCC unroll 15
            for (uint32_t c = 1; c < 16; c++) {
                g[c] = 0u;
            }
            tt_l1_ptr uint32_t* bb = reinterpret_cast<tt_l1_ptr uint32_t*>(cb_b.get_write_ptr());
            const uint32_t b_val = chunk_element(raw_b, bb, tr, gr, b_bf16 != 0);
#pragma GCC unroll 16
            for (uint32_t r = 0; r < 16; r++) {
                bb[cb::off_16x32(r, 0) / 4] = b_val;
            }
#pragma GCC unroll 15
            for (uint32_t c = 1; c < 16; c++) {
                bb[c] = 0u;
            }
        }
        compiler_fence();
        cb_qk.push_back(Kt);
        cb_kT.push_back(Kt);
        cb_qc.push_back(Kt);
        cb_v.push_back(Vt);
        cb_g.push_back(1);
        cb_b.push_back(1);
    };

    // ---- initial state S (once): the caller's state (block h, or ring block idx[h]: caller contract in the Python
    // binding), or zeros replicated from zero32 when the caller gave none. Token 0's reads go out under the same
    // barrier.
    {
        cb_S.reserve_back(kv);
        if constexpr (has_state) {
            uint32_t s0_block = h;
            if constexpr (use_blk_idx) {
                const uint32_t idx_addr = get_arg_val<uint32_t>(8);
                const uint32_t idx_page_bytes = get_arg_val<uint32_t>(9);
                [[maybe_unused]] const uint32_t BH = get_arg_val<uint32_t>(10);
                CircularBuffer cb_idx(cb::blkidx);
                const CoreLocalMem<volatile uint32_t> idx(cb_idx.get_write_ptr());
                const auto idx_acc = TensorAccessor(idx_a, idx_addr, idx_page_bytes);
                noc.async_read(idx_acc, cb_idx, idx_page_bytes, {.page_id = 0}, {.offset_bytes = 0});
                noc.async_read_barrier();
                s0_block = idx[h];
                // Watcher-checked caller contract: the block belongs to head h and lies inside the ring.
                ASSERT(s0_block % BH == h && s0_block < T * BH);
            }
            for (uint32_t i = 0; i < kv; i++) {
                noc.async_read(s0_acc, cb_S, s_tb, {.page_id = s0_block * kv + i}, {.offset_bytes = i * s_tb});
            }
        } else {
            const uint32_t dst = cb_S.get_write_ptr();
            for (uint32_t i = 0; i < kv; i++) {
                noc_async_read(zero_src, dst + i * tb32, tb32);
            }
        }
        issue_token_reads(0);
        noc.async_read_barrier();
        cb_S.push_back(kv);
    }
    finish_token(0);

    // ---- tokens 1.. --------------------------------------------------------------------------------------------
    for (uint32_t t = 1; t < T; t++) {
        {
            issue_token_reads(t);
            noc.async_read_barrier();
        }
        finish_token(t);
    }
}
