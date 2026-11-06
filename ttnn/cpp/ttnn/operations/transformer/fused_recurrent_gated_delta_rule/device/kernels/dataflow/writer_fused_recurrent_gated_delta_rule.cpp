// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer: o per token into the caller's [B, T, HV, V] tensor in the input dtype; the state [K, V] per token (verify
// slots) or once (final), into the caller's layouts. Device 2.0 API.
//
// o arrives as Vt fp32 tiles of 16x32 (row 0 = o; see ../fused_recurrent_gdn_cb.hpp): core (b, hv) writes its two
// face rows (16 elements each) into row hv % 32 of tile ((b*T + t) * ceil(HV/32) + hv/32) * Vt + j of the output, 64 B
// of fp32 or, for a bf16 output, 32 B of bf16 that the RISC rounds (to nearest even) into the obf16 scratch first.
// DRAM writes are 16-byte aligned, so the rows go straight to their place; no other core writes that row.
//
// The state tiles are written as they arrive: the compute packs state_out in (i, j) order, so the writer waits for
// them cumulatively in groups of Vt (one i row) and issues each group's NoC writes at once, instead of waiting for all
// Kt*Vt tiles and then writing 64 KB after the last token. The CB memory is read by the NoC asynchronously, so a group
// is popped only after the token's single write barrier; state_out holds kv x 2 tiles, so the compute can pack the
// next token meanwhile.
//
// Per-token state goes to [B, T, HV, K, V]: slab (b*T + t) * HV + hv, the caller's layout, no host relayout. In ring
// mode (the Qwen3.6 verify path) it goes token-major into the ring: slab t*BH + h, so the [T*BH, K, V] ring reads as
// [T, B, HV, K, V] and the host commits by choosing the next call's block index. The final state goes to slab h of
// [B, HV, K, V], which is also rows [0:B] of the caller's state when the output aliases it (inplace_final_state).
//
// Every runtime argument is common to the cores; the head index is the core's coordinates: h = x * grid_y + y.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

#include "../fused_recurrent_gdn_cb.hpp"

namespace cb = gdn_decode_cb;

namespace {
// Byte offset of face column f of row r in a 32x32 tile of e-byte elements (faces 0..3 are rows 0-15 / 16-31 by
// columns 0-15 / 16-31; a face is 16 x 16 elements).
constexpr uint32_t tile_row_off(uint32_t r, uint32_t f, uint32_t e) {
    return ((r / 16u) * 2u + f) * 256u * e + (r % 16u) * 16u * e;
}
// fp32 bits -> bf16 bits, round to nearest even (a NaN keeps its quiet bit).
inline uint32_t to_bf16(uint32_t w) {
    if ((w & 0x7F800000u) == 0x7F800000u && (w & 0x007FFFFFu) != 0u) {
        return (w >> 16) | 0x40u;
    }
    return (w + 0x7FFFu + ((w >> 16) & 1u)) >> 16;
}
// The compiler must not move the RISC's L1 accesses across a NoC write issue (MMIO).
inline void compiler_fence() { asm volatile("" ::: "memory"); }
}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t per_token = get_compile_time_arg_val(2);
    constexpr uint32_t ring = get_compile_time_arg_val(3);    // per-token state token-major into the ring
    constexpr uint32_t o_bf16 = get_compile_time_arg_val(4);  // o's dtype (the input dtype)
    constexpr uint32_t HV = get_compile_time_arg_val(5);

    constexpr auto o_a = TensorAccessorArgs<6>();
    constexpr auto st_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();

    const uint32_t T = get_common_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t st_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t BH = get_common_arg_val<uint32_t>(3);  // the ring's token-major stride
    const uint32_t grid_y = get_common_arg_val<uint32_t>(4);
    const uint32_t h = get_absolute_logical_x() * grid_y + get_absolute_logical_y();

    // o: 32x32 tiles of the output dtype in DRAM, fp32 16x32 tiles in the CB; the state: 32x32 fp32 tiles.
    constexpr uint32_t e_o = o_bf16 ? 2u : 4u;
    constexpr uint32_t o_page_tb = 1024u * e_o;
    constexpr uint32_t o_row_tb = 16u * e_o;               // one face row of the output: the 16 live elements of a face
    constexpr uint32_t cb_face_tb = cb::face_bytes_16x16;  // the fp32 CB tile's faces
    constexpr uint32_t st_tb = cb::tile_bytes(cb::tile_32x32);
    const uint32_t o_cb_tb = get_tile_size(cb::out16);  // 16x32 fp32, from the CB's declared dims
    const auto o_acc = TensorAccessor(o_a, o_addr, o_page_tb);
    const auto st_acc = TensorAccessor(st_a, st_addr, st_tb);

    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t group = Vt;  // state tiles per wait: one i row of the compute's (i, j) pack order
    constexpr uint32_t HV_t = (HV + 31u) / 32u;
    const uint32_t b = h / HV;
    const uint32_t hv = h % HV;
    const uint32_t vr = hv % 32u, vpr = hv / 32u;  // o's row within the tile, tile row

    Noc noc;
    CircularBuffer cbout(cb::out16);
    CircularBuffer cbst(cb::state_out);
    CircularBuffer cbobf(cb::obf16);
    auto o_src = use<CircularBuffer::AddrSelector::READ_PTR>(cbout);
    auto st_src = use<CircularBuffer::AddrSelector::READ_PTR>(cbst);
    auto obf_src = use<CircularBuffer::AddrSelector::WRITE_PTR>(cbobf);
    if constexpr (o_bf16) {
        cbobf.reserve_back(1);  // writer-private: reserved once, never pushed
    }

    // Token t's o tiles: wait for all Vt (the vector pass pushes them together), write row 0 of each (its two face
    // rows) into row vr of the output tiles of (b, t). For a bf16 output the rows are rounded into the scratch first;
    // the scratch is reused only after the token's write barrier.
    auto issue_o = [&](uint32_t t) {
        cbout.wait_front(Vt);
        const uint32_t page0 = ((b * T + t) * HV_t + vpr) * Vt;
        if constexpr (o_bf16) {
            const tt_l1_ptr uint32_t* __restrict__ src =
                reinterpret_cast<const tt_l1_ptr uint32_t*>(cbout.get_read_ptr());
            tt_l1_ptr uint32_t* __restrict__ dst = reinterpret_cast<tt_l1_ptr uint32_t*>(cbobf.get_write_ptr());
            for (uint32_t j = 0; j < Vt; j++) {
                for (uint32_t f = 0; f < 2; f++) {
                    const tt_l1_ptr uint32_t* row = src + (j * o_cb_tb + f * cb_face_tb) / 4;
                    tt_l1_ptr uint32_t* out = dst + (j * 2 + f) * 8;
#pragma GCC unroll 8
                    for (uint32_t w = 0; w < 8; w++) {
                        out[w] = to_bf16(row[2 * w]) | (to_bf16(row[2 * w + 1]) << 16);
                    }
                }
            }
            compiler_fence();
        }
        for (uint32_t j = 0; j < Vt; j++) {
            for (uint32_t f = 0; f < 2; f++) {
                if constexpr (o_bf16) {
                    noc.async_write(
                        obf_src,
                        o_acc,
                        o_row_tb,
                        {.offset_bytes = (j * 2 + f) * 32u},
                        {.page_id = page0 + j, .offset_bytes = tile_row_off(vr, f, e_o)});
                } else {
                    noc.async_write(
                        o_src,
                        o_acc,
                        o_row_tb,
                        {.offset_bytes = j * o_cb_tb + f * cb_face_tb},
                        {.page_id = page0 + j, .offset_bytes = tile_row_off(vr, f, e_o)});
                }
            }
        }
    };
    // One state: kv tiles into pages base_page.., issued group by group as they become visible (cumulative waits; the
    // read pointer does not move until the pop after the barrier).
    auto issue_state = [&](uint32_t base_page) {
        for (uint32_t g0 = 0; g0 < kv; g0 += group) {
            const uint32_t n = (kv - g0 < group) ? (kv - g0) : group;
            cbst.wait_front(g0 + n);
            for (uint32_t i = 0; i < n; i++) {
                noc.async_write(
                    st_src, st_acc, st_tb, {.offset_bytes = (g0 + i) * st_tb}, {.page_id = base_page + g0 + i});
            }
        }
    };

    for (uint32_t t = 0; t < T; t++) {
        issue_o(t);
        if constexpr (per_token) {
            const uint32_t slab = ring ? (t * BH + h) : ((b * T + t) * HV + hv);
            issue_state(slab * kv);
        }
        noc.async_write_barrier();
        cbout.pop_front(Vt);
        if constexpr (per_token) {
            cbst.pop_front(kv);
        }
    }
    if constexpr (!per_token) {
        issue_state(h * kv);  // final state only: slab h of [B, HV, K, V]
        noc.async_write_barrier();
        cbst.pop_front(kv);
    }
}
