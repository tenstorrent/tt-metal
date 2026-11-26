// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// compute_drain bench reader (one core, one head): W[h] once, then Mt tile rows of attn sticks.
// Each tile row is read in KC K-chunks (KC = KC_LAST on the last row, KC_OTHER on the others): chunk c of
// every stick (bytes [c*cw, (c+1)*cw)) lands at chunk region c (kc tile-pages, rows of cw bytes), so a
// kc-wide tilize can consume it. KC = 1 is the op's reader (whole sticks, one push per row).
// PACING: the push of row r chunk c waits until t0 + r*ROW_GAP + (c+1)*ROW_GAP/KC cycles, emulating the
// contended DRAM landing rate of the real op (~1.9 us per row on the critical cores); the landing times are
// therefore identical for every compute variant and the end-to-end difference is the drain difference.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_weight = get_compile_time_arg_val(0);
    constexpr uint32_t cb_attn_rm = get_compile_time_arg_val(1);
    constexpr uint32_t Kt = get_compile_time_arg_val(2);
    constexpr uint32_t Nt = get_compile_time_arg_val(3);
    constexpr uint32_t Mt = get_compile_time_arg_val(4);
    constexpr uint32_t stick_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t KC_OTHER = get_compile_time_arg_val(7);
    constexpr uint32_t KC_LAST = get_compile_time_arg_val(8);
    constexpr uint32_t ROW_GAP = get_compile_time_arg_val(9);  // cycles
    constexpr auto attn_args = TensorAccessorArgs<10>();
    constexpr auto w_args = TensorAccessorArgs<attn_args.next_compile_time_args_offset()>();

    const uint32_t attn_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const auto attn = TensorAccessor(attn_args, attn_addr, stick_bytes);
    const auto w = TensorAccessor(w_args, w_addr, w_tile_bytes);
    constexpr uint32_t tile_bytes = 2048;  // bf16 tile-page of cb_attn_rm

    cb_reserve_back(cb_weight, Kt * Nt);
    uint32_t l1w = get_write_ptr(cb_weight);
    noc_async_read_set_trid(15);
    for (uint32_t t = 0; t < Kt * Nt; ++t) {
        noc_async_read_page(t, w, l1w);
        l1w += w_tile_bytes;
    }
    const uint64_t t0 = get_timestamp();

    for (uint32_t r = 0; r < Mt; ++r) {
        const uint32_t KC = (r == Mt - 1) ? KC_LAST : KC_OTHER;
        const uint32_t kc = Kt / KC;
        const uint32_t cw = stick_bytes / KC;
        cb_reserve_back(cb_attn_rm, Kt);
        const uint32_t base = get_write_ptr(cb_attn_rm);
        for (uint32_t c = 0; c < KC; ++c) {
            noc_async_read_set_trid(1 + c);
            for (uint32_t i = 0; i < 32; ++i) {
                noc_async_read(attn.get_noc_addr(r * 32 + i, c * cw), base + c * kc * tile_bytes + i * cw, cw);
            }
        }
        for (uint32_t c = 0; c < KC; ++c) {
            noc_async_read_barrier_with_trid(1 + c);
            const uint64_t target = t0 + (uint64_t)r * ROW_GAP + ((uint64_t)(c + 1) * ROW_GAP) / KC;
            while (get_timestamp() < target) {
            }
            if (r == Mt - 1 && c == KC - 1) {
                DeviceZoneScopedN("cd_last_landed");
                cb_push_back(cb_attn_rm, kc);
            } else {
                cb_push_back(cb_attn_rm, kc);
            }
        }
        if (r == 0) {
            noc_async_read_barrier_with_trid(15);
            cb_push_back(cb_weight, Kt * Nt);
        }
    }
    noc_async_read_set_trid(0);
}
