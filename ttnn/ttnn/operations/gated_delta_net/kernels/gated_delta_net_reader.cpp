// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gated_delta_net reader (NCRISC).
//
// Per pair (b, h) of this core's contiguous range:
//   load_pair_block: conv tap tiles (row j of conv_weight -> row 0 of tap tile j), A_log / dt_bias
//                    gate-column tiles, norm_weight tiles.
//   per tile row of the pair: b/a gate tiles, the conv window (4 tap tiles per column tile, rows
//                    outside the pair's token rows zero), S0 after the first tile row's prep
//                    inputs (stall shadow), z tiles.
//   copy_conv_state_block: conv_state_out rows 3b..3b+2 of the pair's columns (bit-exact DM copy).
// Token rows are moved as 64 B face-row segments unless the pair owns the whole tile row.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {

constexpr uint32_t cb_state_in = 0;
constexpr uint32_t cb_state_in_mm = 1;
constexpr uint32_t cb_conv_w = 5;
constexpr uint32_t cb_window = 6;
constexpr uint32_t cb_gate_in = 14;
constexpr uint32_t cb_gate_w = 15;
constexpr uint32_t cb_z = 21;
constexpr uint32_t cb_norm_w = 22;
constexpr uint32_t cb_reduce_scaler = 24;
constexpr uint32_t cb_scratch = 27;

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t TILE_BYTES = 4096;  // fp32 tile
constexpr uint32_t SEG_BYTES = 64;     // one fp32 face row
constexpr uint32_t FACE_BYTES = 1024;  // one fp32 16x16 face
constexpr uint32_t CONV_TAPS = 4;
constexpr uint32_t HISTORY = 3;

// Byte offset of the left half of tile row p (faces 0/2); the right half is FACE_BYTES further.
FORCE_INLINE uint32_t row_offset(uint32_t p) { return (p >> 4) * 2 * FACE_BYTES + (p & 15) * SEG_BYTES; }

template <typename Accessor>
FORCE_INLINE void read_row(const Accessor& acc, uint32_t page, uint32_t src_row, uint32_t dst_tile, uint32_t dst_row) {
    const uint32_t src_off = row_offset(src_row);
    const uint32_t dst = dst_tile + row_offset(dst_row);
    noc_async_read(acc.get_noc_addr(page, src_off), dst, SEG_BYTES);
    noc_async_read(acc.get_noc_addr(page, src_off + FACE_BYTES), dst + FACE_BYTES, SEG_BYTES);
}

FORCE_INLINE void copy_local_row(uint32_t src_tile, uint32_t src_row, uint32_t dst_tile, uint32_t dst_row) {
    const uint32_t src = src_tile + row_offset(src_row);
    const uint32_t dst = dst_tile + row_offset(dst_row);
    noc_async_read(get_noc_addr(src), dst, SEG_BYTES);
    noc_async_read(get_noc_addr(src + FACE_BYTES), dst + FACE_BYTES, SEG_BYTES);
}

template <typename Accessor>
FORCE_INLINE void write_row(const Accessor& acc, uint32_t src_tile, uint32_t src_row, uint32_t page, uint32_t dst_row) {
    const uint32_t src = src_tile + row_offset(src_row);
    const uint32_t dst_off = row_offset(dst_row);
    noc_async_write(src, acc.get_noc_addr(page, dst_off), SEG_BYTES);
    noc_async_write(src + FACE_BYTES, acc.get_noc_addr(page, dst_off + FACE_BYTES), SEG_BYTES);
}

// Rows [lo, hi) of tile row tr (global row indices) of one tile column into an L1 tile.
template <typename Accessor>
FORCE_INLINE void read_tile_rows(
    const Accessor& acc, uint32_t page, uint32_t tile_row_base, uint32_t lo, uint32_t hi, uint32_t dst) {
    if (hi - lo == TILE_ROWS) {
        noc_async_read(acc.get_noc_addr(page), dst, TILE_BYTES);
    } else {
        for (uint32_t row = lo; row < hi; ++row) {
            read_row(acc, page, row - tile_row_base, dst, row - tile_row_base);
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t Hv = get_compile_time_arg_val(2);
    constexpr uint32_t Hk = get_compile_time_arg_val(3);
    constexpr uint32_t Ct = get_compile_time_arg_val(4);
    constexpr uint32_t Gt = get_compile_time_arg_val(5);
    constexpr uint32_t ratio = Hv / Hk;
    constexpr uint32_t n_state = Kt * Vt;
    constexpr uint32_t z_width_tiles = Hv * Vt;
    constexpr uint32_t pair_cols = 2 * Kt + Vt;

    constexpr auto mixed_args = TensorAccessorArgs<6>();
    constexpr auto z_args = TensorAccessorArgs<mixed_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();
    constexpr auto a_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto conv_state_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto state_args = TensorAccessorArgs<conv_state_args.next_compile_time_args_offset()>();
    constexpr auto conv_w_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    constexpr auto a_log_args = TensorAccessorArgs<conv_w_args.next_compile_time_args_offset()>();
    constexpr auto dt_bias_args = TensorAccessorArgs<a_log_args.next_compile_time_args_offset()>();
    constexpr auto norm_w_args = TensorAccessorArgs<dt_bias_args.next_compile_time_args_offset()>();
    constexpr auto conv_state_out_args = TensorAccessorArgs<norm_w_args.next_compile_time_args_offset()>();

    uint32_t arg = 0;
    const auto mixed = TensorAccessor(mixed_args, get_arg_val<uint32_t>(arg++), TILE_BYTES);
    const auto z_acc = TensorAccessor(z_args, get_arg_val<uint32_t>(arg++), TILE_BYTES);
    const auto b_acc = TensorAccessor(b_args, get_arg_val<uint32_t>(arg++), TILE_BYTES);
    const auto a_acc = TensorAccessor(a_args, get_arg_val<uint32_t>(arg++), TILE_BYTES);
    const auto conv_state = TensorAccessor(conv_state_args, get_arg_val<uint32_t>(arg++), TILE_BYTES);
    const auto state = TensorAccessor(state_args, get_arg_val<uint32_t>(arg++), TILE_BYTES);
    const auto conv_w = TensorAccessor(conv_w_args, get_arg_val<uint32_t>(arg++), TILE_BYTES);
    const auto a_log = TensorAccessor(a_log_args, get_arg_val<uint32_t>(arg++), TILE_BYTES);
    const auto dt_bias = TensorAccessor(dt_bias_args, get_arg_val<uint32_t>(arg++), TILE_BYTES);
    const auto norm_w = TensorAccessor(norm_w_args, get_arg_val<uint32_t>(arg++), TILE_BYTES);
    const auto conv_state_out = TensorAccessor(conv_state_out_args, get_arg_val<uint32_t>(arg++), TILE_BYTES);
    const uint32_t first_pair = get_arg_val<uint32_t>(arg++);
    const uint32_t num_pairs = get_arg_val<uint32_t>(arg++);
    const uint32_t T = get_arg_val<uint32_t>(arg++);

    Noc noc;
    CircularBuffer window_buf(cb_window);

    dataflow_kernel_lib::
        prepare_reduce_scaler<cb_reduce_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(1.0f);

    for (uint32_t pair = first_pair; pair < first_pair + num_pairs; ++pair) {
        const uint32_t seq = pair / Hv;
        const uint32_t head = pair % Hv;
        const uint32_t key_head = head / ratio;
        // Column tiles of the pair in channel-group order: q (Kt), k (Kt), v (Vt).
        auto pair_col = [&](uint32_t idx) -> uint32_t {
            if (idx < Kt) {
                return key_head * Kt + idx;
            }
            if (idx < 2 * Kt) {
                return Hk * Kt + key_head * Kt + (idx - Kt);
            }
            return 2 * Hk * Kt + head * Vt + (idx - 2 * Kt);
        };

        // ---- load_pair_block ----
        cb_reserve_back(cb_conv_w, CONV_TAPS * pair_cols);
        cb_reserve_back(cb_gate_w, 2);
        cb_reserve_back(cb_norm_w, Vt);
        const uint32_t conv_w_base = get_write_ptr(cb_conv_w);
        for (uint32_t idx = 0; idx < pair_cols; ++idx) {
            const uint32_t ct = pair_col(idx);
            for (uint32_t tap = 0; tap < CONV_TAPS; ++tap) {
                read_row(conv_w, ct, tap, conv_w_base + (CONV_TAPS * idx + tap) * TILE_BYTES, 0);
            }
        }
        const uint32_t gate_w_base = get_write_ptr(cb_gate_w);
        noc_async_read(a_log.get_noc_addr(head / TILE_ROWS), gate_w_base, TILE_BYTES);
        noc_async_read(dt_bias.get_noc_addr(head / TILE_ROWS), gate_w_base + TILE_BYTES, TILE_BYTES);
        const uint32_t norm_w_base = get_write_ptr(cb_norm_w);
        for (uint32_t vt = 0; vt < Vt; ++vt) {
            noc_async_read(norm_w.get_noc_addr(vt), norm_w_base + vt * TILE_BYTES, TILE_BYTES);
        }
        noc_async_read_barrier();
        cb_push_back(cb_conv_w, CONV_TAPS * pair_cols);
        cb_push_back(cb_gate_w, 2);
        cb_push_back(cb_norm_w, Vt);

        const uint32_t seq_lo = seq * T;
        const uint32_t seq_hi = seq_lo + T;
        const uint32_t first_tr = seq_lo / TILE_ROWS;
        const uint32_t last_tr = (seq_hi - 1) / TILE_ROWS;
        for (uint32_t tr = first_tr; tr <= last_tr; ++tr) {
            const uint32_t base_row = tr * TILE_ROWS;
            const uint32_t lo = seq_lo > base_row ? seq_lo : base_row;
            const uint32_t hi = seq_hi < base_row + TILE_ROWS ? seq_hi : base_row + TILE_ROWS;
            const bool owned = (hi - lo) == TILE_ROWS;

            // ---- gate inputs: b, a tiles of the gate column ----
            cb_reserve_back(cb_gate_in, 2);
            const uint32_t gate_in_base = get_write_ptr(cb_gate_in);
            const uint32_t gate_page = tr * Gt + head / TILE_ROWS;
            noc_async_read(b_acc.get_noc_addr(gate_page), gate_in_base, TILE_BYTES);
            noc_async_read(a_acc.get_noc_addr(gate_page), gate_in_base + TILE_BYTES, TILE_BYTES);
            noc_async_read_barrier();
            cb_push_back(cb_gate_in, 2);

            // ---- conv window: X_j[p] = xx[row(p) + j - 3], one column tile per hand-off ----
            for (uint32_t idx = 0; idx < pair_cols; ++idx) {
                const uint32_t ct = pair_col(idx);
                cb_reserve_back(cb_window, CONV_TAPS);
                const uint32_t win = get_write_ptr(cb_window);
                const uint32_t cur_tap = win + (CONV_TAPS - 1) * TILE_BYTES;
                if (!owned) {
                    // Rows outside [lo, hi) are other sequences or padding: zero, never sourced.
                    noc.async_write_zeros(window_buf, CONV_TAPS * TILE_BYTES, {.offset_bytes = 0});
                    noc.write_zeros_l1_barrier();
                }
                read_tile_rows(mixed, tr * Ct + ct, base_row, lo, hi, cur_tap);
                noc_async_read_barrier();
                for (uint32_t tap = 0; tap < CONV_TAPS - 1; ++tap) {
                    const uint32_t tap_tile = win + tap * TILE_BYTES;
                    for (uint32_t row = lo; row < hi; ++row) {
                        const uint32_t p = row - base_row;
                        const uint32_t shift = HISTORY - tap;  // source row = row - shift
                        if (row >= lo + shift) {
                            copy_local_row(cur_tap, p - shift, tap_tile, p);
                        } else if (row >= seq_lo + shift) {
                            const uint32_t src = row - shift;
                            read_row(mixed, (src / TILE_ROWS) * Ct + ct, src % TILE_ROWS, tap_tile, p);
                        } else {
                            // History row n = t + tap of sequence seq (t = row - seq_lo).
                            const uint32_t src = HISTORY * seq + (row - seq_lo) + tap;
                            read_row(conv_state, (src / TILE_ROWS) * Ct + ct, src % TILE_ROWS, tap_tile, p);
                        }
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_window, CONV_TAPS);
            }

            // ---- S0 of the pair, after the first tile row's prep inputs ----
            if (tr == first_tr) {
                cb_reserve_back(cb_state_in, n_state);
                cb_reserve_back(cb_state_in_mm, n_state);
                const uint32_t state_base = get_write_ptr(cb_state_in);
                for (uint32_t i = 0; i < n_state; ++i) {
                    noc_async_read(state.get_noc_addr(pair * n_state + i), state_base + i * TILE_BYTES, TILE_BYTES);
                }
                noc_async_read_barrier();
                cb_push_back(cb_state_in, n_state);
                cb_push_back(cb_state_in_mm, n_state);
            }

            // ---- z columns of the head ----
            cb_reserve_back(cb_z, Vt);
            const uint32_t z_base = get_write_ptr(cb_z);
            for (uint32_t vt = 0; vt < Vt; ++vt) {
                read_tile_rows(z_acc, tr * z_width_tiles + head * Vt + vt, base_row, lo, hi, z_base + vt * TILE_BYTES);
            }
            noc_async_read_barrier();
            cb_push_back(cb_z, Vt);
        }

        // ---- copy_conv_state_block: conv_state_out[3b + n] = xx[b, T + n] ----
        const uint32_t scratch = get_write_ptr(cb_scratch);
        const uint32_t first_idx = (head % ratio == 0) ? 0 : 2 * Kt;  // q/k columns written by one head per key head
        for (uint32_t idx = first_idx; idx < pair_cols; ++idx) {
            const uint32_t ct = pair_col(idx);
            for (uint32_t n = 0; n < HISTORY; ++n) {
                if (T + n < HISTORY) {
                    const uint32_t src = HISTORY * seq + T + n;
                    read_row(conv_state, (src / TILE_ROWS) * Ct + ct, src % TILE_ROWS, scratch, n);
                } else {
                    const uint32_t src = seq_lo + T + n - HISTORY;
                    read_row(mixed, (src / TILE_ROWS) * Ct + ct, src % TILE_ROWS, scratch, n);
                }
            }
            noc_async_read_barrier();
            for (uint32_t n = 0; n < HISTORY; ++n) {
                const uint32_t dst = HISTORY * seq + n;
                write_row(conv_state_out, scratch, n, (dst / TILE_ROWS) * Ct + ct, dst % TILE_ROWS);
            }
            noc_async_write_barrier();
        }
    }
}
