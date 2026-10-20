// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader of the fused KDA decode block. Per work item (head h, user tile row tb): the raw q/k/v tile
// blocks of the head (bf16), the three conv-history rows of every user of the row (bf16 RM, 256 B each,
// tilized by compute) -- and, having read them, shifts the history in place (row j+1 -> row j; the
// writer appends the raw rows as row 2) --, the taps, the gate blocks, the channel constants and then,
// per user, the recurrent state. The selector tiles and the o_norm weight are read once per core.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_raw = 0, cb_hist = 1, cb_taps = 2, cb_gf = 3, cb_gog = 4, cb_gb = 5, cb_cc = 6, cb_onw = 7;
constexpr uint32_t cb_sel = 8, cb_S = 9;

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t Bt = get_compile_time_arg_val(2);
    constexpr uint32_t B = get_compile_time_arg_val(3);
    constexpr uint32_t H = get_compile_time_arg_val(4);
    constexpr uint32_t Wq_t = get_compile_time_arg_val(5);   // qkv row width in tiles (3*H*Kt)
    constexpr uint32_t Wg_t = get_compile_time_arg_val(6);   // gates row width in tiles
    constexpr uint32_t HK_t = get_compile_time_arg_val(7);   // H*Kt
    constexpr uint32_t HV_t = get_compile_time_arg_val(8);   // H*Vt
    constexpr uint32_t parts = get_compile_time_arg_val(9);  // user split of a tile row over cores
    constexpr uint32_t row_bytes = Kt * 32 * 2;              // one head's bf16 row of a branch (K elements)
    constexpr uint32_t conv_taps = 3;                        // history rows (kernel size 4)

    constexpr auto qkv_a = TensorAccessorArgs<10>();
    constexpr auto conv_a = TensorAccessorArgs<qkv_a.next_compile_time_args_offset()>();
    constexpr auto taps_a = TensorAccessorArgs<conv_a.next_compile_time_args_offset()>();
    constexpr auto gates_a = TensorAccessorArgs<taps_a.next_compile_time_args_offset()>();
    constexpr auto cc_a = TensorAccessorArgs<gates_a.next_compile_time_args_offset()>();
    constexpr auto onw_a = TensorAccessorArgs<cc_a.next_compile_time_args_offset()>();
    constexpr auto sel_a = TensorAccessorArgs<onw_a.next_compile_time_args_offset()>();
    constexpr auto st_a = TensorAccessorArgs<sel_a.next_compile_time_args_offset()>();

    const uint32_t item0 = get_arg_val<uint32_t>(0);
    const uint32_t n_items = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);
    const uint32_t qkv_addr = get_arg_val<uint32_t>(3);
    const uint32_t conv_addr = get_arg_val<uint32_t>(4);
    const uint32_t taps_addr = get_arg_val<uint32_t>(5);
    const uint32_t gates_addr = get_arg_val<uint32_t>(6);
    const uint32_t cc_addr = get_arg_val<uint32_t>(7);
    const uint32_t onw_addr = get_arg_val<uint32_t>(8);
    const uint32_t sel_addr = get_arg_val<uint32_t>(9);
    const uint32_t st_addr = get_arg_val<uint32_t>(10);
    const uint32_t conv_page_bytes = get_arg_val<uint32_t>(11);  // one [b, j] row of the conv state

    const uint32_t t32 = get_tile_size(cb_S);    // fp32 tile
    const uint32_t t16 = get_tile_size(cb_raw);  // bf16 tile
    const auto qkv_acc = TensorAccessor(qkv_a, qkv_addr, t16);
    const auto conv_acc = TensorAccessor(conv_a, conv_addr, conv_page_bytes);
    const auto taps_acc = TensorAccessor(taps_a, taps_addr, t32);
    const auto gates_acc = TensorAccessor(gates_a, gates_addr, t32);
    const auto cc_acc = TensorAccessor(cc_a, cc_addr, t32);
    const auto onw_acc = TensorAccessor(onw_a, onw_addr, t32);
    const auto sel_acc = TensorAccessor(sel_a, sel_addr, t16);
    const auto st_acc = TensorAccessor(st_a, st_addr, t32);

    Noc noc;
    auto read_tiles = [&](const auto& acc, uint32_t cb_id, uint32_t tb, uint32_t base, uint32_t n) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n);
        for (uint32_t t = 0; t < n; t++) {
            noc.async_read(acc, cb, tb, {.page_id = base + t}, {.offset_bytes = t * tb});
        }
        noc.async_read_barrier();
        cb.push_back(n);
    };

    // once per core: o_norm weight (row 0 of its tile row) and the 33 selector tiles
    read_tiles(onw_acc, cb_onw, t32, 0, Vt);
    read_tiles(sel_acc, cb_sel, t16, 0, 33);
    // the history RM blocks are reused with the same L1 addresses: zero them once so rows beyond the
    // live users of a short tile row are finite (they feed the selector matmuls as 0 * x)
    {
        CircularBuffer cb(cb_hist);
        const CoreLocalMem<volatile uint32_t> p(cb.get_write_ptr());
        for (uint32_t i = 0; i < conv_taps * 3 * Kt * t16 / 4; i++) {
            p[i] = 0;
        }
    }

    for (uint32_t it = 0; it < n_items; it++) {
        const uint32_t item = item0 + it * stride;
        const uint32_t h = item / (Bt * parts);
        const uint32_t rem = item % (Bt * parts);
        const uint32_t tb = rem / parts;
        const uint32_t part = rem % parts;
        const uint32_t n_row = (B - tb * 32u < 32u) ? (B - tb * 32u) : 32u;
        const uint32_t up = (n_row + parts - 1) / parts;
        const uint32_t u0 = part * up;
        const uint32_t u1 = (u0 + up < n_row) ? (u0 + up) : n_row;
        if (u0 >= u1) {
            continue;
        }

        // taps of the three branches (tile row 0 of [4, 3*H*K]): branch br, tile i at page br*HK_t + h*Kt + i
        {
            CircularBuffer cb(cb_taps);
            cb.reserve_back(3 * Kt);
            for (uint32_t br = 0; br < 3; br++) {
                for (uint32_t i = 0; i < Kt; i++) {
                    noc.async_read(
                        taps_acc, cb, t32, {.page_id = br * HK_t + h * Kt + i}, {.offset_bytes = (br * Kt + i) * t32});
                }
            }
            noc.async_read_barrier();
            cb.push_back(3 * Kt);
        }
        // per branch: the raw tiles and the three history blocks (shifted in place right after reading)
        for (uint32_t br = 0; br < 3; br++) {
            read_tiles(qkv_acc, cb_raw, t16, tb * Wq_t + br * HK_t + h * Kt, Kt);
            const uint32_t col_off = (br * HK_t + h * Kt) * 32 * 2;  // byte offset of the head's row inside a conv page
            for (uint32_t j = 0; j < conv_taps; j++) {
                CircularBuffer cb(cb_hist);
                cb.reserve_back(Kt);
                for (uint32_t r = u0; r < u1; r++) {
                    const uint32_t page = (tb * 32 + r) * conv_taps + j;
                    noc.async_read(
                        conv_acc,
                        cb,
                        row_bytes,
                        {.page_id = page, .offset_bytes = col_off},
                        {.offset_bytes = r * row_bytes});
                }
                noc.async_read_barrier();
                if (j > 0) {
                    // history shift: row j becomes row j-1 (the writer appends the raw rows as row 2). The rows
                    // were just read to the CB's WRITE side (not yet pushed), so write them from there.
                    auto src = use<CircularBuffer::AddrSelector::WRITE_PTR>(cb);
                    for (uint32_t r = u0; r < u1; r++) {
                        const uint32_t page = (tb * 32 + r) * conv_taps + (j - 1);
                        noc.async_write(
                            src,
                            conv_acc,
                            row_bytes,
                            {.offset_bytes = r * row_bytes},
                            {.page_id = page, .offset_bytes = col_off});
                    }
                    noc.async_write_barrier();
                }
                cb.push_back(Kt);
            }
        }
        // gate blocks: f [32, K] at (tb, h*Kt), og at (tb, HK_t + h*Vt), beta logits tile at (tb, HK_t + HV_t + h/32)
        read_tiles(gates_acc, cb_gf, t32, tb * Wg_t + h * Kt, Kt);
        read_tiles(cc_acc, cb_cc, t32, h * Kt, Kt);
        read_tiles(gates_acc, cb_gb, t32, tb * Wg_t + HK_t + HV_t + (h >> 5), 1);
        read_tiles(gates_acc, cb_gog, t32, tb * Wg_t + HK_t + h * Vt, Vt);
        // per user: the recurrent state block (b*H + h)
        for (uint32_t r = u0; r < u1; r++) {
            read_tiles(st_acc, cb_S, t32, ((tb * 32 + r) * H + h) * Kt * Vt, Kt * Vt);
        }
    }
}
