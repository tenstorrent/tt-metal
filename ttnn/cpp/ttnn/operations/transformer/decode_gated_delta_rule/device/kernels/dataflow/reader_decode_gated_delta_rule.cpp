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
// input (q,k,v,beta,g,state). Runtime args: {bh, q,k,v,beta,g,state addrs}.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// CB indices (must match program factory / compute).
constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4;
constexpr uint32_t cb_state = 5, cb_ones = 6, cb_scratch = 27;

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

    const uint32_t bh = get_arg_val<uint32_t>(0);
    const uint32_t q_addr = get_arg_val<uint32_t>(1);
    const uint32_t k_addr = get_arg_val<uint32_t>(2);
    const uint32_t v_addr = get_arg_val<uint32_t>(3);
    const uint32_t beta_addr = get_arg_val<uint32_t>(4);
    const uint32_t g_addr = get_arg_val<uint32_t>(5);
    const uint32_t s0_addr = get_arg_val<uint32_t>(6);

    const uint32_t tb_io = get_tile_size(cb_q);   // q/k/v/g/beta/state share one dtype
    const uint32_t elem = tb_io / 1024;           // bytes per element
    const auto q_acc = TensorAccessor(q_a, q_addr, tb_io);
    const auto k_acc = TensorAccessor(k_a, k_addr, tb_io);
    const auto v_acc = TensorAccessor(v_a, v_addr, tb_io);
    const auto beta_acc = TensorAccessor(beta_a, beta_addr, tb_io);
    const auto g_acc = TensorAccessor(g_a, g_addr, tb_io);
    const auto s0_acc = TensorAccessor(s0_a, s0_addr, tb_io);

    constexpr uint32_t kv = Kt * Vt;

    Noc noc;

    // This head's row inside the shared 32-row tile pages, in face coords.
    const uint32_t r = bh % 32;
    const uint32_t frow = r % 16;   // row within a face
    const uint32_t fhalf = r / 16;  // 0: rows 0-15 (faces 0,1), 1: rows 16-31 (faces 2,3)
    // Source element offsets (within a DRAM page) of cols 0-15 / 16-31 of row r:
    const uint32_t src_e0 = (fhalf * 2 + 0) * 256 + frow * 16;
    const uint32_t src_e1 = (fhalf * 2 + 1) * 256 + frow * 16;
    // Destination element offsets for row 0 of the private page (faces 0 and 1):
    constexpr uint32_t dst_e0 = 0;
    constexpr uint32_t dst_e1 = 256;

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

    // Core-side L1 copy of one 16-element face-row chunk (8 words). The source
    // words were written by NOC DMA: volatile reads + a compiler barrier keep
    // the loads after the DMA barrier (core_local_mem.h ordering note).
    auto copy_chunk = [&](uint32_t src_bytes, uint32_t dst_bytes) {
        asm volatile("" ::: "memory");
        auto s = CoreLocalMem<volatile uint32_t>(src_bytes);
        auto d = CoreLocalMem<volatile uint32_t>(dst_bytes);
        for (uint32_t w = 0; w < 8; w++) {
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

    // Gather head bh's single row (n_tiles tiles of D=32 each) out of the
    // staged shared pages into row 0 of n_tiles private CB pages.
    auto gather_row = [&](const auto& acc, uint32_t cb_id, uint32_t n_tiles) {
        read_pages(acc, (bh / 32) * n_tiles, n_tiles);
        CircularBuffer scb(cb_scratch);
        scb.wait_front(n_tiles);
        const uint32_t sbase = scb.get_read_ptr();
        CircularBuffer cb(cb_id);
        cb.reserve_back(n_tiles);
        const uint32_t base = cb.get_write_ptr();
        for (uint32_t t = 0; t < n_tiles; t++) {
            zero(base + t * tb_io, tb_io / 4);
        }
        for (uint32_t t = 0; t < n_tiles; t++) {
            const uint32_t src = sbase + t * tb_io;
            const uint32_t dst = base + t * tb_io;
            copy_chunk(src + src_e0 * elem, dst + dst_e0 * elem);
            copy_chunk(src + src_e1 * elem, dst + dst_e1 * elem);
        }
        scb.pop_front(n_tiles);
        cb.push_back(n_tiles);
    };

    // Gather the single [B,1,H] head-scalar of head bh=(b,h): the [B,H] flat
    // 2D puts it at (row b, col h) — page b/32, row b%32, col h. Copy the one
    // element (word load, 16-bit select for bf16) into [0,0] of a zeroed tile.
    auto gather_scalar = [&](const auto& acc, uint32_t cb_id) {
        const uint32_t b = bh / H;
        const uint32_t hcol = bh % H;
        read_pages(acc, b / 32, 1);
        CircularBuffer scb(cb_scratch);
        scb.wait_front(1);
        const uint32_t sbase = scb.get_read_ptr();
        CircularBuffer cb(cb_id);
        cb.reserve_back(1);
        const uint32_t base = cb.get_write_ptr();
        zero(base, tb_io / 4);
        // element (row b%32, col hcol) offset within the tile page
        const uint32_t rr = b % 32;
        const uint32_t eoff =
            ((rr / 16) * 2 + (hcol / 16)) * 256 + (rr % 16) * 16 + (hcol % 16);
        const uint32_t byte = sbase + eoff * elem;
        asm volatile("" ::: "memory");
        auto s = CoreLocalMem<volatile uint32_t>(byte & ~3u);
        auto d = CoreLocalMem<volatile uint32_t>(base);
        uint32_t w = s[0];
        if (elem == 2 && (byte & 2u)) {
            w >>= 16;  // bf16 scalar sits in the high half of this word
        }
        d[0] = w;  // [0,0] of the private tile = low half of word 0
        asm volatile("" ::: "memory");
        scb.pop_front(1);
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

    gather_row(q_acc, cb_q, Kt);
    gather_row(k_acc, cb_k, Kt);
    gather_row(v_acc, cb_v, Vt);
    gather_scalar(beta_acc, cb_beta);
    gather_scalar(g_acc, cb_g);

    // State [B,H,K,V] flat 2D is [B*H*K, V]: head bh owns full row-tiles
    // (bh*Kt .. bh*Kt+Kt), so its kv pages are contiguous.
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
