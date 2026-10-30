// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader: initial state S [K,V] once (from s0 or host-provided zeros), then per token the tiny-tile operands of the
// two-pass recurrence (see ../fused_recurrent_gdn_cb.hpp for the plan). All fp32. Device 2.0 API.
//
// Source layout (host-prepared, today's convention): q, k [BH*T, 1, K], v [BH*T, 1, V] and g, beta [BH*T, 1, 1] are
// fp32 TILE tensors, i.e. 32x32 tiles with only row 0 live (g and beta: only element [0, 0]); block index = h*T + t.
// Only the live rows are read: the two 64-byte face rows of row 0 (tile bytes 0-63 = columns 0-15, bytes 1024-1087 =
// columns 16-31) go straight into the CB tiles, and the column tiles (kT16, qcol32, gcol32, bcol16) are filled by a
// RISC loop from the landed row data. The zero paddings of every token CB are written once at kernel start: the CBs
// hold token_cb_nbuf slot sets that the per-token reserve/push walks in a fixed order, so a padding is never
// overwritten (at T = 1 only the first slot set is used and zeroed).
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

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t use_blk_idx = get_compile_time_arg_val(2);

    constexpr auto q_a = TensorAccessorArgs<3>();
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

    // Every source tensor is 32x32 fp32 tiles (4 KB pages); the CB tiles are the tiny ones of the plan.
    constexpr uint32_t src_tb = cb::tile_bytes(cb::tile_32x32);
    const auto q_acc = TensorAccessor(q_a, q_addr, src_tb);
    const auto k_acc = TensorAccessor(k_a, k_addr, src_tb);
    const auto v_acc = TensorAccessor(v_a, v_addr, src_tb);
    const auto g_acc = TensorAccessor(g_a, g_addr, src_tb);
    const auto b_acc = TensorAccessor(b_a, b_addr, src_tb);
    const auto s0_acc = TensorAccessor(s0_a, s0_addr, src_tb);

    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t tb16 = cb::tile_bytes(cb::tile_16x32);  // 2048: also the 32x16 tile
    constexpr uint32_t tb32 = cb::tile_bytes(cb::tile_32x32);  // 4096
    constexpr uint32_t row = cb::row_bytes;                    // 64: one 16-element face row
    // Source byte offsets of the live row 0 of a 32x32 tile: columns 0-15 (face 0) and 16-31 (face 1).
    constexpr uint32_t src_row0_f0 = cb::off_32x32_row(0, 0);
    constexpr uint32_t src_row0_f1 = cb::off_32x32_row(0, 1);

    Noc noc;
    CircularBuffer cb_qk(cb::qk16), cb_kT(cb::kT16), cb_qc(cb::qcol32), cb_v(cb::v16);
    CircularBuffer cb_g(cb::gcol32), cb_b(cb::bcol16), cb_ones(cb::ones32), cb_zero(cb::zero32);
    // The state is one allocation under two ids (utd for the compute's exact copies, mm for its matmul); both are
    // reserved and pushed here so their pointers stay in lockstep.
    CircularBuffer cb_S(cb::S), cb_S_mm(cb::S_mm);

    // ---- once at kernel start: constants and the zero paddings of every per-token CB ----------------------------
    // Reserving the whole CB (nothing is pushed yet) gives the base of all its slot sets; the per-token reserve/push
    // below walks them in a fixed order. zero32 is zeroed by the NoC's zero write; every other padding is a local NoC
    // copy of zero32 in 4 KB pieces (the zero write itself goes in small loopback pieces: 96 KB of it cost 3.7 us).
    {
        constexpr uint32_t nb = cb::token_cb_nbuf;
        const uint32_t nb_used = T > 1 ? nb : 1u;  // slot sets the token loop will touch
        cb_zero.reserve_back(1);
        noc.async_write_zeros(cb_zero, tb32);
        noc.write_zeros_l1_barrier();
        const uint64_t zero_src = get_noc_addr(cb_zero.get_write_ptr());
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
    auto issue_token_reads = [&](uint32_t t) {
        const uint32_t block = h * T + t;
        cb_qk.reserve_back(Kt);
        cb_kT.reserve_back(Kt);
        cb_qc.reserve_back(Kt);
        cb_v.reserve_back(Vt);
        cb_g.reserve_back(1);
        cb_b.reserve_back(1);
        // qk16[ki]: row 0 = k (face 0 and face 1 rows 0), row 2 = q.
        for (uint32_t ki = 0; ki < Kt; ki++) {
            const uint32_t page = block * Kt + ki;
            const uint32_t dst = ki * tb16;
            noc.async_read(
                k_acc,
                cb_qk,
                row,
                {.page_id = page, .offset_bytes = src_row0_f0},
                {.offset_bytes = dst + cb::off_16x32(0, 0)});
            noc.async_read(
                k_acc,
                cb_qk,
                row,
                {.page_id = page, .offset_bytes = src_row0_f1},
                {.offset_bytes = dst + cb::off_16x32(0, 1)});
            noc.async_read(
                q_acc,
                cb_qk,
                row,
                {.page_id = page, .offset_bytes = src_row0_f0},
                {.offset_bytes = dst + cb::off_16x32(2, 0)});
            noc.async_read(
                q_acc,
                cb_qk,
                row,
                {.page_id = page, .offset_bytes = src_row0_f1},
                {.offset_bytes = dst + cb::off_16x32(2, 1)});
        }
        // v16[j]: row 0 = v.
        for (uint32_t j = 0; j < Vt; j++) {
            const uint32_t page = block * Vt + j;
            const uint32_t dst = j * tb16;
            noc.async_read(
                v_acc,
                cb_v,
                row,
                {.page_id = page, .offset_bytes = src_row0_f0},
                {.offset_bytes = dst + cb::off_16x32(0, 0)});
            noc.async_read(
                v_acc,
                cb_v,
                row,
                {.page_id = page, .offset_bytes = src_row0_f1},
                {.offset_bytes = dst + cb::off_16x32(0, 1)});
        }
        // g, beta: the live row 0 of face 0 is [value, 0 x 15]; it lands as row 0 of the column tile, and finish_token
        // replicates element [0, 0] down column 0.
        noc.async_read(g_acc, cb_g, row, {.page_id = block, .offset_bytes = src_row0_f0}, {.offset_bytes = 0});
        noc.async_read(b_acc, cb_b, row, {.page_id = block, .offset_bytes = src_row0_f0}, {.offset_bytes = 0});
    };

    // Call after the reads of issue_token_reads have landed (a NoC read barrier).
    auto finish_token = [&]() {
        compiler_fence();
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
            tt_l1_ptr uint32_t* g = reinterpret_cast<tt_l1_ptr uint32_t*>(cb_g.get_write_ptr());
            const uint32_t g_val = g[0];
#pragma GCC unroll 31
            for (uint32_t r = 1; r < 32; r++) {
                g[cb::off_32x32_col0(r) / 4] = g_val;
            }
            tt_l1_ptr uint32_t* b = reinterpret_cast<tt_l1_ptr uint32_t*>(cb_b.get_write_ptr());
            const uint32_t b_val = b[0];
#pragma GCC unroll 15
            for (uint32_t r = 1; r < 16; r++) {
                b[cb::off_16x32(r, 0) / 4] = b_val;
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

    // ---- initial state S (once) -- the host always provides it (zeros if absent). In ring mode the block is idx[h]
    // instead of h (caller contract: see the Python binding). Token 0's reads go out under the same barrier.
    {
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
        cb_S.reserve_back(kv);
        cb_S_mm.reserve_back(kv);
        for (uint32_t i = 0; i < kv; i++) {
            noc.async_read(s0_acc, cb_S, src_tb, {.page_id = s0_block * kv + i}, {.offset_bytes = i * src_tb});
        }
        issue_token_reads(0);
        noc.async_read_barrier();
        cb_S.push_back(kv);
        cb_S_mm.push_back(kv);
    }
    finish_token();

    // ---- tokens 1.. --------------------------------------------------------------------------------------------
    for (uint32_t t = 1; t < T; t++) {
        {
            issue_token_reads(t);
            noc.async_read_barrier();
        }
        finish_token();
    }
}
