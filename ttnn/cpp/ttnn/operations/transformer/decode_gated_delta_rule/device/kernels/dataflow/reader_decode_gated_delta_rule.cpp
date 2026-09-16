// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader for the fused T=1 decode gated delta rule. Device 2.0 API.
//
// T=1 inputs [B,1,H,*] are TILE tensors whose flat 2D view is [B*H, D]: 32
// heads share each row of TILE pages. This core owns head bh; its row lives at
// row (bh % 32) of the shared pages. TILE pages are four 16x16 faces in
// row-major face order (face_idx = (r/16)*2 + (c/16), elem offset
// face_idx*256 + (r%16)*16 + (c%16)).
//
// Dataflow rule (learned the hard way): noc reads must be FULL tile-size,
// page-aligned accessor reads — sub-page reads (any size, either via accessor
// offset_bytes or a raw tensor_accessor::Page address) silently fail and leave
// the zero-filled CB pages untouched. So: full pages are DMA'd into a staging
// CB, then the head's row is extracted with core-side L1 word copies into the
// private row-0 CB pages. (Sub-page WRITES fail the same way — see the writer.)
//
// Scalar tensors beta/g are [B,1,H]: their flat 2D is [B,H], so head (b,h)'s
// scalar lives at ROW b, COLUMN h of the shared page — NOT at (bh, 0). Reading
// (bh,0) was the red-state bug: every head but (0,0) gathered zero padding
// (beta=0, g=0 => state returned unchanged, pcc_h ~0.97).
//
// The state [B,H,K,V] is head-aligned (K%32==0): full-tile reads. A fp32
// all-ones tile is synthesized for the compute row-sum.
//
// Compile args: {Kt, Vt, has_s0, eps_bits, scale_bits, H} + accessor args per
// input (q,k,v,beta,g,state). Runtime args: {bh_start, n_inst, q,k,v,beta,g,
// state addrs}: this core loops over its contiguous instance chunk
// [bh_start, bh_start+n_inst) (BH = B*H can exceed the core count).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// CB indices (must match program factory / compute).
constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4;
constexpr uint32_t cb_state = 5, cb_ones = 6, cb_scratch = 27;
// Destination element offsets for row 0 of a private page (faces 0 and 1).
constexpr uint32_t dst_e0 = 0;
constexpr uint32_t dst_e1 = 256;

// TILE physical layout: the last two dims of each outer slice pad to 32.
// q/k/v [B,1,H,D]: head (b,h) lives at physical row b*padH + h (padH =
// ceil32(H)), NOT at flat row b*H + h — equal only when B == 1 or H%32 == 0
// (the original bug: batch>0 gathered per-batch padding zeros / the wrong
// head's row). beta/g [B,1,H]: dims (1,H) pad per batch: (b,hcol) sits at
// page b*ppb + hcol/32, ROW 0, col hcol%32 (ppb = padH/32 pages/batch).
// state [B,H,K,V]: K,V are 32-multiples — flat contiguous, head bh at bh*kv.

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t has_s0 = get_compile_time_arg_val(2);
    constexpr uint32_t H = get_compile_time_arg_val(5);  // heads (bh = b*H + h)

    constexpr auto q_a = TensorAccessorArgs<6>();
    constexpr auto k_a = TensorAccessorArgs<q_a.next_compile_time_args_offset()>();
    constexpr auto v_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    constexpr auto beta_a = TensorAccessorArgs<v_a.next_compile_time_args_offset()>();
    constexpr auto g_a = TensorAccessorArgs<beta_a.next_compile_time_args_offset()>();
    constexpr auto s0_a = TensorAccessorArgs<g_a.next_compile_time_args_offset()>();

    // Runtime args (factory order): {bh_start(0), n_inst(1), q(2), k(3), v(4),
    // beta(5), g(6), state(7)}: this core loops over its contiguous instance
    // chunk [bh_start, bh_start+n_inst) (BH = B*H can exceed the core count).
    const uint32_t bh_start = get_arg_val<uint32_t>(0);
    const uint32_t n_inst = get_arg_val<uint32_t>(1);
    const uint32_t q_addr = get_arg_val<uint32_t>(2);
    const uint32_t k_addr = get_arg_val<uint32_t>(3);
    const uint32_t v_addr = get_arg_val<uint32_t>(4);
    const uint32_t beta_addr = get_arg_val<uint32_t>(5);
    const uint32_t g_addr = get_arg_val<uint32_t>(6);
    const uint32_t s0_addr = get_arg_val<uint32_t>(7);

    const uint32_t tb_io = get_tile_size(cb_q);   // q/k/v/g/beta/state share one dtype
    const uint32_t elem = tb_io / 1024;           // bytes per element
    const auto q_acc = TensorAccessor(q_a, q_addr, tb_io);
    const auto k_acc = TensorAccessor(k_a, k_addr, tb_io);
    const auto v_acc = TensorAccessor(v_a, v_addr, tb_io);
    const auto beta_acc = TensorAccessor(beta_a, beta_addr, tb_io);
    const auto g_acc = TensorAccessor(g_a, g_addr, tb_io);
    const auto s0_acc = TensorAccessor(s0_a, s0_addr, tb_io);

    constexpr uint32_t kv = Kt * Vt;
    // Per-batch padded head-row geometry (see the layout note above).
    const uint32_t padH = ((H + 31) / 32) * 32;
    const uint32_t ppb = padH / 32;  // beta/g pages per batch

    Noc noc;

    // Zero an L1 region of n words, then return. Pages are zeroed so padded
    // rows/cols are exact zeros (the rank-1 outer product multiplies through
    // padding and must produce exact zeros, matching the python TILE graph).
    // Volatile: these words are later overwritten by NOC DMA / read back after
    // DMA — non-volatile accesses can be reordered by the compiler (see the
    // note in core_local_mem.h).
    auto zero = [&](uint32_t base, uint32_t n_words) {
        auto ptr = CoreLocalMem<volatile uint32_t>(base);
        for (uint32_t w = 0; w < n_words; w++) {
            ptr[w] = 0u;
        }
        asm volatile("" ::: "memory");
    };

    // Core-side L1 copy of one 16-element face-row chunk (16*elem/4 words:
    // 8 at bf16, 16 at fp32 — a fixed word count silently dropped half the
    // chunk at fp32). The source words were written by NOC DMA: volatile
    // reads + a compiler barrier keep the loads after the DMA barrier
    // (core_local_mem.h ordering note).
    auto copy_chunk = [&](uint32_t src_bytes, uint32_t dst_bytes) {
        asm volatile("" ::: "memory");
        auto s = CoreLocalMem<volatile uint32_t>(src_bytes);
        auto d = CoreLocalMem<volatile uint32_t>(dst_bytes);
        for (uint32_t w = 0; w < 16 * elem / 4; w++) {
            d[w] = s[w];
        }
        asm volatile("" ::: "memory");
    };

    // DMA n_tiles FULL pages (page-aligned, tile-size reads — the only proven
    // read shape) into the staging CB; caller then extracts the head's row.
    auto read_pages = [&](const auto& acc, uint32_t first_page, uint32_t n_tiles) {
        CircularBuffer cb(cb_scratch);
        cb.reserve_back(n_tiles);
        for (uint32_t t = 0; t < n_tiles; t++) {
            noc.async_read(acc, cb, tb_io, {.page_id = first_page + t}, {.offset_bytes = t * tb_io});
        }
        noc.async_read_barrier();
        cb.push_back(n_tiles);
    };

    // Gather the row at PHYSICAL row `physrow` (each tile-row spans n_tiles
    // pages of D=32 columns): DMA the shared tile pages DIRECTLY into the
    // target CB (the proven full-page read shape — the scratch-staged variant
    // deadlocked at fp32 with multiple instances per core), then extract row
    // r into row 0 IN PLACE (row r and row 0 are distinct rows; r == 0 is a
    // no-op) and zero every other row so the tile is row-0-only as the compute
    // expects.
    auto gather_row = [&](const auto& acc, uint32_t cb_id, uint32_t n_tiles, uint32_t physrow) {
        const uint32_t first_page = (physrow / 32) * n_tiles;
        const uint32_t r = physrow % 32;
        const uint32_t frow = r % 16;   // row within a face
        const uint32_t fhalf = r / 16;  // 0: rows 0-15 (faces 0,1), 1: rows 16-31 (faces 2,3)
        const uint32_t src_e0 = (fhalf * 2 + 0) * 256 + frow * 16;
        const uint32_t src_e1 = (fhalf * 2 + 1) * 256 + frow * 16;
        CircularBuffer cb(cb_id);
        cb.reserve_back(n_tiles);
        const uint32_t base = cb.get_write_ptr();
        for (uint32_t t = 0; t < n_tiles; t++) {
            noc.async_read(acc, cb, tb_io, {.page_id = first_page + t}, {.offset_bytes = t * tb_io});
        }
        noc.async_read_barrier();
        for (uint32_t t = 0; t < n_tiles; t++) {
            const uint32_t p = base + t * tb_io;
            if (r != 0) {
                copy_chunk(p + src_e0 * elem, p);                  // row 0 cols 0-15
                copy_chunk(p + src_e1 * elem, p + 256 * elem);     // row 0 cols 16-31
            }
            // zero everything except row 0's two 16-element face chunks
            zero(p + 16 * elem, (256 - 16) * elem / 4);
            zero(p + (256 + 16) * elem, (1024 - 272) * elem / 4);
        }
        cb.push_back(n_tiles);
    };

    // Gather the single [B,1,H] head-scalar of head bh=(b,h): (b,hcol) sits
    // at page b*ppb + hcol/32, ROW 0, col hcol%32 of the per-batch padded
    // tiles (layout note above). DMA the full tile page DIRECTLY into the
    // target CB page (the proven full-page read shape; no scratch staging),
    // then select the one element (word load, 16-bit select for bf16) into
    // [0,0] of the same zeroed page.
    auto gather_scalar = [&](const auto& acc, uint32_t cb_id, uint32_t bh) {
        const uint32_t b = bh / H;
        const uint32_t hcol = bh % H;
        CircularBuffer cb(cb_id);
        cb.reserve_back(1);
        const uint32_t base = cb.get_write_ptr();
        zero(base, tb_io / 4);
        noc.async_read(acc, cb, tb_io, {.page_id = b * ppb + hcol / 32}, {.offset_bytes = 0});
        noc.async_read_barrier();
        const uint32_t cc = hcol % 32;
        const uint32_t eoff = (cc / 16) * 256 + (cc % 16);
        const uint32_t byte = base + eoff * elem;
        asm volatile("" ::: "memory");
        auto s = CoreLocalMem<volatile uint32_t>(byte & ~3u);
        auto d = CoreLocalMem<volatile uint32_t>(base);
        uint32_t w = s[0];
        if (elem == 2 && (byte & 2u)) {
            w >>= 16;  // bf16 scalar sits in the high half of this word
        }
        d[0] = w;  // [0,0] of the private tile = low half of word 0
        asm volatile("" ::: "memory");
        cb.push_back(1);
    };

    // fp32 all-ones tile for the compute kernel's row-sum contraction.
    {
        CircularBuffer cb(cb_ones);
        cb.reserve_back(1);
        const uint32_t base = cb.get_write_ptr();
        auto ptr = CoreLocalMem<uint32_t>(base);
        for (uint32_t w = 0; w < 1024; w++) {
            ptr[w] = 0x3F800000u;  // fp32 1.0
        }
        cb.push_back(1);
    }

    // Per-instance loop: this core owns the contiguous chunk
    // [bh_start, bh_start + n_inst) of head-instances.
    for (uint32_t bh = bh_start; bh < bh_start + n_inst; ++bh) {
        // Head (b,h)'s physical row in the [B,1,H,D] TILE tensors.
        const uint32_t physrow = (bh / H) * padH + (bh % H);

        gather_row(q_acc, cb_q, Kt, physrow);
        gather_row(k_acc, cb_k, Kt, physrow);
        gather_row(v_acc, cb_v, Vt, physrow);
        gather_scalar(beta_acc, cb_beta, bh);
        gather_scalar(g_acc, cb_g, bh);

        // State [B,H,K,V] flat 2D is [B*H*K, V] (K,V are 32-multiples: no
        // per-batch padding): head bh owns full row-tiles (bh*Kt .. bh*Kt+Kt).
        CircularBuffer cbs(cb_state);
        cbs.reserve_back(kv);
        if (has_s0) {
            const uint32_t base_page = bh * kv;
            for (uint32_t t = 0; t < kv; t++) {
                noc.async_read(
                    s0_acc, cbs, tb_io, {.page_id = base_page + t}, {.offset_bytes = t * tb_io});
            }
            noc.async_read_barrier();
        } else {
            zero(cbs.get_write_ptr(), kv * tb_io / 4);
        }
        cbs.push_back(kv);
    }
}
