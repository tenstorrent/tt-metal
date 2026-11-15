// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gated_delta_net writer (BRISC).
//
// Per pair, per tile row: expands the pair's (alpha, beta) gate elements into three column-0
// coefficient pages per token (alpha in every row; beta at row p; alpha at row p), then drains
// the tile row's output tiles (whole tiles when the pair owns the tile row, else face-row
// segments of its rows only). After the pair's last token: drains recurrent_state_out.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

// CB indices and GDN_* spec constants are host defines (gated_delta_net_program_descriptor.py).
constexpr uint32_t cb_state_out = GDN_CB_STATE_OUT;
constexpr uint32_t cb_gates = GDN_CB_GATES;
constexpr uint32_t cb_tok_coef = GDN_CB_TOK_COEF;
constexpr uint32_t cb_out_rows = GDN_CB_OUT_ROWS;

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t OUT_TILE_BYTES = get_compile_time_arg_val(3);    // output tile (dtype-derived on host)
constexpr uint32_t STATE_TILE_BYTES = get_compile_time_arg_val(4);  // fp32 state tile
constexpr uint32_t FACE_BYTES = OUT_TILE_BYTES / 4;
constexpr uint32_t SEG_BYTES = FACE_BYTES / 16;
constexpr uint32_t TILE_WORDS = 1024;  // cb_gates / cb_tok_coef are Float32 compute buffers

FORCE_INLINE uint32_t row_offset(uint32_t p) { return (p >> 4) * 2 * FACE_BYTES + (p & 15) * SEG_BYTES; }

// Output face-row segments are written directly: Tensix-to-DRAM writes need only L1 alignment.
// Word index of element (r, c) of an fp32 tile (four 16x16 faces).
FORCE_INLINE uint32_t elem_index(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t Hv = get_compile_time_arg_val(2);
    constexpr uint32_t n_state = Kt * Vt;
    constexpr uint32_t out_width_tiles = Hv * Vt;

    constexpr auto out_args = TensorAccessorArgs<5>();
    constexpr auto state_out_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    uint32_t arg = 0;
    const auto out_acc = TensorAccessor(out_args, get_arg_val<uint32_t>(arg++), OUT_TILE_BYTES);
    const auto state_out = TensorAccessor(state_out_args, get_arg_val<uint32_t>(arg++), STATE_TILE_BYTES);
    const uint32_t first_pair = get_arg_val<uint32_t>(arg++);
    const uint32_t num_pairs = get_arg_val<uint32_t>(arg++);
    const uint32_t T = get_arg_val<uint32_t>(arg++);

    for (uint32_t pair = first_pair; pair < first_pair + num_pairs; ++pair) {
        const uint32_t seq = pair / Hv;
        const uint32_t head = pair % Hv;
        const uint32_t gate_col = head % TILE_ROWS;
        const uint32_t seq_lo = seq * T;
        const uint32_t seq_hi = seq_lo + T;
        const uint32_t first_tr = seq_lo / TILE_ROWS;
        const uint32_t last_tr = (seq_hi - 1) / TILE_ROWS;

        for (uint32_t tr = first_tr; tr <= last_tr; ++tr) {
            const uint32_t base_row = tr * TILE_ROWS;
            const uint32_t lo = seq_lo > base_row ? seq_lo : base_row;
            const uint32_t hi = seq_hi < base_row + TILE_ROWS ? seq_hi : base_row + TILE_ROWS;

            // ---- per-token coefficient expansion ----
            cb_wait_front(cb_gates, 2);
            volatile tt_l1_ptr uint32_t* gates = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_gates));
            for (uint32_t row = lo; row < hi; ++row) {
                const uint32_t p = row - base_row;
                const uint32_t alpha = gates[elem_index(p, gate_col)];
                const uint32_t beta = gates[TILE_WORDS + elem_index(p, gate_col)];
                cb_reserve_back(cb_tok_coef, GDN_TOK_COEF_PAGES);
                volatile tt_l1_ptr uint32_t* coef =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_tok_coef));
                for (uint32_t r = 0; r < TILE_ROWS; ++r) {
                    const uint32_t e = elem_index(r, 0);
                    const bool at_p = (r == p);
                    coef[e] = alpha;
                    coef[TILE_WORDS + e] = at_p ? beta : 0u;
                    coef[2 * TILE_WORDS + e] = at_p ? alpha : 0u;
                }
                cb_push_back(cb_tok_coef, GDN_TOK_COEF_PAGES);
            }
            cb_pop_front(cb_gates, 2);

            // ---- store_rows_block ----
            cb_wait_front(cb_out_rows, Vt);
            const uint32_t out_base = get_read_ptr(cb_out_rows);
            for (uint32_t vt = 0; vt < Vt; ++vt) {
                const uint32_t page = tr * out_width_tiles + head * Vt + vt;
                const uint32_t src_tile = out_base + vt * OUT_TILE_BYTES;
                if (hi - lo == TILE_ROWS) {
                    noc_async_write(src_tile, out_acc.get_noc_addr(page), OUT_TILE_BYTES);
                } else {
                    for (uint32_t row = lo; row < hi; ++row) {
                        const uint32_t off = row_offset(row - base_row);
                        noc_async_write(src_tile + off, out_acc.get_noc_addr(page, off), SEG_BYTES);
                        noc_async_write(
                            src_tile + off + FACE_BYTES, out_acc.get_noc_addr(page, off + FACE_BYTES), SEG_BYTES);
                    }
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out_rows, Vt);
        }

        // ---- store_state_block ----
        cb_wait_front(cb_state_out, n_state);
        const uint32_t state_base = get_read_ptr(cb_state_out);
        for (uint32_t i = 0; i < n_state; ++i) {
            noc_async_write(
                state_base + i * STATE_TILE_BYTES, state_out.get_noc_addr(pair * n_state + i), STATE_TILE_BYTES);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_state_out, n_state);
    }
}
