// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gated_delta_net writer (BRISC).
//
// Per unit (pair, piece), per tile row: expands the pair's (alpha, beta) gate elements into three
// column-0 coefficient pages per token (alpha in every row; beta at row p; alpha at row p); with
// PIECES > 1 (R-VSPLIT) all-gathers the partial sum o^2 of every piece of the pair; then drains the
// piece's output tiles (whole tiles when the pair owns the tile row, else face-row segments of its rows
// only). After the pair's last token: drains the piece's slice of recurrent_state_out.
// R-BCAST (decode R-VSPLIT): the key head's q and k injectors multicast qn and kn (one channel each) into
// cb_kq_rows on every core of the key head, where the writer hands the landed rows to compute.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

namespace {

// CB indices and GDN_* spec constants are host defines (gated_delta_net_program_descriptor.py).
constexpr uint32_t cb_state_out = GDN_CB_STATE_OUT;
constexpr uint32_t cb_gates = GDN_CB_GATES;
constexpr uint32_t cb_tok_coef = GDN_CB_TOK_COEF;
constexpr uint32_t cb_out_rows = GDN_CB_OUT_ROWS;
constexpr uint32_t cb_window = GDN_CB_WINDOW;
constexpr uint32_t cb_window_zeroed = GDN_CB_WINDOW_ZEROED;
constexpr uint32_t cb_osum_part = GDN_CB_OSUM_PART;
constexpr uint32_t cb_osum_gather = GDN_CB_OSUM_GATHER;
constexpr uint32_t cb_kq_rows = GDN_CB_KQ_ROWS;
constexpr uint32_t cb_kq_rows_f32 = GDN_CB_KQ_ROWS_F32;
constexpr uint32_t cb_qk_stage = GDN_CB_QK_STAGE;

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t OUT_TILE_BYTES = get_compile_time_arg_val(3);    // output tile (dtype-derived on host)
constexpr uint32_t STATE_TILE_BYTES = get_compile_time_arg_val(4);  // fp32 state tile
constexpr uint32_t FACE_BYTES = OUT_TILE_BYTES / 4;
constexpr uint32_t SEG_BYTES = FACE_BYTES / 16;
constexpr uint32_t PIECES = get_compile_time_arg_val(5);       // column pieces per pair
constexpr uint32_t OSUM_SEM_ID = get_compile_time_arg_val(6);  // peer partial sums landed (cumulative)
constexpr bool CHUNKED = get_compile_time_arg_val(7) != 0;     // R-CHUNK: chunk coefficients per tile row
constexpr uint32_t TILE_WORDS = 1024;  // cb_gates / cb_tok_coef are Float32 compute buffers

FORCE_INLINE uint32_t row_offset(uint32_t p) { return (p >> 4) * 2 * FACE_BYTES + (p & 15) * SEG_BYTES; }

// Output face-row segments are written directly: Tensix-to-DRAM writes need only L1 alignment.
// Word index of element (r, c) of an fp32 tile (four 16x16 faces).
FORCE_INLINE uint32_t elem_index(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

// Lands kn (cb_kq_rows tiles [0, Kt)) and qn ([Kt, 2Kt)) of one tile row. An injector multicasts its staged
// part (qn from the q injector, kn from the k injector), its own rows included (the sender lies inside its
// group, so a send from another address loops back and is complete when send() returns); every core receives
// the parts it did not prep. Every core has the same CB layout and R-BCAST hands off one block per core, so a
// receiver's write pointer is the address the injector targets.
template <typename QChannel, typename KChannel>
void land_qk_rows(const QChannel& q_channel, const KChannel& k_channel, uint32_t qk_role, uint32_t Kt) {
    Noc noc;
    const uint32_t part_bytes = Kt * STATE_TILE_BYTES;
    cb_reserve_back(cb_kq_rows, 2 * Kt);
    cb_reserve_back(cb_kq_rows_f32, 2 * Kt);
    const uint32_t kn_dst = get_write_ptr(cb_kq_rows);
    const uint32_t qn_dst = kn_dst + part_bytes;
    if (qk_role == GDN_QK_Q_INJECTOR || qk_role == GDN_QK_K_INJECTOR) {
        const bool is_q = qk_role == GDN_QK_Q_INJECTOR;
        const uint32_t dst = is_q ? qn_dst : kn_dst;
        cb_wait_front(cb_qk_stage, Kt);
        const uint32_t stage = get_read_ptr(cb_qk_stage);
        if (is_q) {
            q_channel.optional_sender(noc)->send(stage, dst, part_bytes);
        } else {
            k_channel.optional_sender(noc)->send(stage, dst, part_bytes);
        }
        cb_pop_front(cb_qk_stage, Kt);
        if (is_q) {
            k_channel.optional_receiver(noc)->receive();
        } else {
            q_channel.optional_receiver(noc)->receive();
        }
    } else {
        q_channel.optional_receiver(noc)->receive();
        k_channel.optional_receiver(noc)->receive();
    }
    cb_push_back(cb_kq_rows, 2 * Kt);
    cb_push_back(cb_kq_rows_f32, 2 * Kt);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t Hv = get_compile_time_arg_val(2);
    constexpr uint32_t Vb = Vt / PIECES;  // v / state column tiles of one piece
    constexpr uint32_t n_state = Kt * Vb;
    constexpr uint32_t out_width_tiles = Hv * Vt;

    constexpr auto out_args = TensorAccessorArgs<8>();
    constexpr auto q_channel = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("qkq_ct_offset"), get_named_compile_time_arg_val("qkq_rt_offset")>();
    constexpr auto k_channel = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("qkk_ct_offset"), get_named_compile_time_arg_val("qkk_rt_offset")>();
    constexpr auto state_out_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    uint32_t arg = 0;
    const auto out_acc = TensorAccessor(out_args, get_arg_val<uint32_t>(arg++), OUT_TILE_BYTES);
    const auto state_out = TensorAccessor(state_out_args, get_arg_val<uint32_t>(arg++), STATE_TILE_BYTES);
    const uint32_t first_unit = get_arg_val<uint32_t>(arg++);
    const uint32_t num_units = get_arg_val<uint32_t>(arg++);
    const uint32_t T = get_arg_val<uint32_t>(arg++);
    const uint32_t qk_role = get_arg_val<uint32_t>(arg++);
    const uint32_t zero_slots = get_arg_val<uint32_t>(arg++);       // cb_window slots this core's reader uses
    const uint32_t zero_slot_bytes = get_arg_val<uint32_t>(arg++);  // one window group
    // R-VSPLIT: NoC (x, y) of the pair's piece cores in piece order (this core included). The host runs
    // one unit per core then, so every piece of a pair walks the same tile rows in lockstep.
    uint32_t peer_x[PIECES];
    uint32_t peer_y[PIECES];
    for (uint32_t s = 0; s < PIECES && PIECES > 1; ++s) {
        peer_x[s] = get_arg_val<uint32_t>(arg++);
        peer_y[s] = get_arg_val<uint32_t>(arg++);
    }
    uint32_t rows_combined = 0;

    // Zero the window groups the reader will use once (it keeps rows outside the token rows zero from then
    // on), one slot at a time with a token per slot so the reader starts on slot 0 without waiting for
    // the rest; this RISC is idle until the first gates arrive.
    {
        Noc noc;
        CircularBuffer window_buf(cb_window);
        for (uint32_t slot = 0; slot < zero_slots; ++slot) {
            noc.async_write_zeros(window_buf, zero_slot_bytes, {.offset_bytes = slot * zero_slot_bytes});
            noc.write_zeros_l1_barrier();
            cb_reserve_back(cb_window_zeroed, 1);
            cb_push_back(cb_window_zeroed, 1);
        }
    }

    for (uint32_t unit = first_unit; unit < first_unit + num_units; ++unit) {
        const uint32_t pair = unit / PIECES;
        const uint32_t piece = unit % PIECES;
        const uint32_t v_first = piece * Vb;
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

            // ---- per-token coefficient expansion (R-CHUNK: per-tile-row g / beta columns) ----
            cb_wait_front(cb_gates, 2);
            volatile tt_l1_ptr uint32_t* gates = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_gates));
            if constexpr (CHUNKED) {
                // Column 0 of page 0: g of the pair, of page 1: beta; exactly 0 on rows that are no token of
                // the pair (no decay, no write), whatever the gate tile holds there.
                cb_reserve_back(cb_tok_coef, GDN_CHUNK_COEF_PAGES);
                volatile tt_l1_ptr uint32_t* coef =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_tok_coef));
                for (uint32_t p = 0; p < TILE_ROWS; ++p) {
                    const bool valid = p + base_row >= lo && p + base_row < hi;
                    const uint32_t e = elem_index(p, 0);
                    coef[e] = valid ? gates[elem_index(p, gate_col)] : 0u;
                    coef[TILE_WORDS + e] = valid ? gates[TILE_WORDS + elem_index(p, gate_col)] : 0u;
                }
                cb_push_back(cb_tok_coef, GDN_CHUNK_COEF_PAGES);
            }
            for (uint32_t row = lo; row < hi && !CHUNKED; ++row) {
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

            if (!CHUNKED && qk_role != GDN_QK_OWN) {
                land_qk_rows(q_channel, k_channel, qk_role, Kt);
            }

            // ---- partial sum o^2 all-gather (R-VSPLIT) ----
            // Each piece writes its partial into slot `piece` of the gather buffer on every piece core
            // (same L1 address everywhere: identical CB layout, identical tile-row count), then bumps
            // the peers' counters. cb_osum_gather holds two tile rows: a peer can be at most one tile
            // row ahead, because it cannot finish row i+1 without this core's partial of row i+1, which
            // compute produces only after it has consumed row i's slots.
            if constexpr (PIECES > 1) {
                Noc noc;
                Semaphore<> osum_landed(OSUM_SEM_ID);
                cb_wait_front(cb_osum_part, 1);
                cb_reserve_back(cb_osum_gather, PIECES);
                const uint32_t part = get_read_ptr(cb_osum_part);
                const uint32_t slot = get_write_ptr(cb_osum_gather) + piece * STATE_TILE_BYTES;
                for (uint32_t s = 0; s < PIECES; ++s) {
                    noc_async_write(part, get_noc_addr(peer_x[s], peer_y[s], slot), STATE_TILE_BYTES);
                }
                noc_async_write_barrier();
                for (uint32_t s = 0; s < PIECES; ++s) {
                    if (s != piece) {
                        osum_landed.up(noc, peer_x[s], peer_y[s], 1);
                    }
                }
                cb_pop_front(cb_osum_part, 1);
                ++rows_combined;
                osum_landed.wait_min(rows_combined * (PIECES - 1));
                cb_push_back(cb_osum_gather, PIECES);
            }

            // ---- store_rows_block ----
            cb_wait_front(cb_out_rows, Vb);
            const uint32_t out_base = get_read_ptr(cb_out_rows);
            for (uint32_t vt = 0; vt < Vb; ++vt) {
                const uint32_t page = tr * out_width_tiles + head * Vt + v_first + vt;
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
            cb_pop_front(cb_out_rows, Vb);
        }

        // ---- store_state_block ----
        cb_wait_front(cb_state_out, n_state);
        const uint32_t state_base = get_read_ptr(cb_state_out);
        for (uint32_t i = 0; i < n_state; ++i) {
            const uint32_t page = pair * Kt * Vt + (i / Vb) * Vt + v_first + i % Vb;
            noc_async_write(state_base + i * STATE_TILE_BYTES, state_out.get_noc_addr(page), STATE_TILE_BYTES);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_state_out, n_state);
    }
}
