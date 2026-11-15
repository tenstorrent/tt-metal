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
// R-PREP prep core (ROLE): expands the coefficient pages of its key head's r value heads per tile row, then, once
// every state core of the group has granted the row's inbox slot, writes the prepped row into their inboxes and
// bumps their landed counters. A state core's writer skips the coefficient expansion (its gates come prepped).

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
constexpr uint32_t cb_prep_send = GDN_CB_PREP_SEND;
constexpr uint32_t cb_knT = GDN_CB_KNT;
constexpr uint32_t cb_v = GDN_CB_V;
constexpr uint32_t cb_chunk_vec = GDN_CB_CHUNK_VEC;
constexpr uint32_t cb_chunk_mat = GDN_CB_CHUNK_MAT;
constexpr uint32_t cb_o_acc = GDN_CB_O_ACC;
constexpr uint32_t cb_o_in = GDN_CB_O_IN;

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t OUT_TILE_BYTES = get_compile_time_arg_val(3);    // output tile (dtype-derived on host)
constexpr uint32_t STATE_TILE_BYTES = get_compile_time_arg_val(4);  // fp32 state tile
constexpr uint32_t FACE_BYTES = OUT_TILE_BYTES / 4;
constexpr uint32_t SEG_BYTES = FACE_BYTES / 16;
constexpr uint32_t PIECES = get_compile_time_arg_val(5);       // column pieces per pair
constexpr uint32_t OSUM_SEM_ID = get_compile_time_arg_val(6);  // peer partial sums landed (cumulative)
constexpr bool CHUNKED = get_compile_time_arg_val(7) != 0;     // R-CHUNK: chunk coefficients per tile row
constexpr uint32_t ROLE = get_compile_time_arg_val(8);         // GDN_ROLE_*
constexpr uint32_t INBOX_SLOTS = get_compile_time_arg_val(9) > 0 ? get_compile_time_arg_val(9)
                                                                 : 1;  // R-PREP inbox slots (1 outside R-PREP)
constexpr uint32_t RATIO = get_compile_time_arg_val(10);               // value heads per key head
constexpr bool PREP = ROLE == GDN_ROLE_PREP;
constexpr bool STATE = ROLE == GDN_ROLE_STATE;
constexpr uint32_t TILE_WORDS = 1024;  // cb_gates / cb_tok_coef are Float32 compute buffers

FORCE_INLINE uint32_t row_offset(uint32_t p) { return (p >> 4) * 2 * FACE_BYTES + (p & 15) * SEG_BYTES; }

// Output face-row segments are written directly: Tensix-to-DRAM writes need only L1 alignment.
// Word index of element (r, c) of an fp32 tile (four 16x16 faces).
FORCE_INLINE uint32_t elem_index(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

// store_rows_block: n output tiles from cb_out_rows to pages first_page.. of tile row [base_row, +32), whole tiles
// when [lo, hi) is the whole tile row, else face-row segments of those rows only.
template <typename Accessor>
void store_rows(const Accessor& out_acc, uint32_t n, uint32_t first_page, uint32_t base_row, uint32_t lo, uint32_t hi) {
    cb_wait_front(cb_out_rows, n);
    const uint32_t out_base = get_read_ptr(cb_out_rows);
    for (uint32_t vt = 0; vt < n; ++vt) {
        const uint32_t page = first_page + vt;
        const uint32_t src_tile = out_base + vt * OUT_TILE_BYTES;
        if (hi - lo == TILE_ROWS) {
            noc_async_write(src_tile, out_acc.get_noc_addr(page), OUT_TILE_BYTES);
        } else {
            for (uint32_t row = lo; row < hi; ++row) {
                const uint32_t off = row_offset(row - base_row);
                noc_async_write(src_tile + off, out_acc.get_noc_addr(page, off), SEG_BYTES);
                noc_async_write(src_tile + off + FACE_BYTES, out_acc.get_noc_addr(page, off + FACE_BYTES), SEG_BYTES);
            }
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out_rows, n);
}

// One multicast write per rectangle of the runtime-arg list at `arg` (count, then x_lo, y_lo, x_hi, y_hi, cores each);
// the sender lies outside every rectangle. NOC 1 takes the corners in the opposite order.
FORCE_INLINE void mcast_rects(uint32_t arg, uint32_t src, uint32_t dst, uint32_t bytes) {
    const uint32_t rects = get_arg_val<uint32_t>(arg++);
    for (uint32_t r = 0; r < rects; ++r, arg += 5) {
        const uint32_t x_lo = get_arg_val<uint32_t>(arg), y_lo = get_arg_val<uint32_t>(arg + 1);
        const uint32_t x_hi = get_arg_val<uint32_t>(arg + 2), y_hi = get_arg_val<uint32_t>(arg + 3);
        const uint64_t dst_noc = noc_index == 0 ? get_noc_multicast_addr(x_lo, y_lo, x_hi, y_hi, dst)
                                                : get_noc_multicast_addr(x_hi, y_hi, x_lo, y_lo, dst);
        noc_async_write_multicast(src, dst_noc, bytes, get_arg_val<uint32_t>(arg + 4));
    }
}

// Column-0 chunk coefficient pages of one pair for tile row [base_row, +32): g (page 0) and beta (page 1) of gate
// column gate_col, exactly 0 on rows outside [lo, hi) (no decay, no write), whatever the gate tile holds there.
FORCE_INLINE void push_chunk_coefs(
    volatile tt_l1_ptr uint32_t* gates, uint32_t gate_col, uint32_t base_row, uint32_t lo, uint32_t hi) {
    cb_reserve_back(cb_tok_coef, GDN_CHUNK_COEF_PAGES);
    volatile tt_l1_ptr uint32_t* coef = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_tok_coef));
    for (uint32_t p = 0; p < TILE_ROWS; ++p) {
        const bool valid = p + base_row >= lo && p + base_row < hi;
        const uint32_t e = elem_index(p, 0);
        coef[e] = valid ? gates[elem_index(p, gate_col)] : 0u;
        coef[TILE_WORDS + e] = valid ? gates[TILE_WORDS + elem_index(p, gate_col)] : 0u;
    }
    cb_push_back(cb_tok_coef, GDN_CHUNK_COEF_PAGES);
}

// R-PREP prep core. Unit = (sequence, key head); tile rows i = offset, offset + stride, ... of the sequence. The row's
// prepped tiles arrive in cb_prep_send as [v of the r heads (r Vt)][kn, qn (2 Kt)][kn^T (Kt)][head pack x r] and go
// to the group's state cores (unit order: head j, piece p) at inbox slot i % INBOX_SLOTS of cb_v (the piece's Vb v
// tiles), cb_kq_rows, cb_knT (all), cb_chunk_vec, cb_chunk_mat (the head's pieces). Every core has the same CB
// layout, so this core's base addresses of those buffers are the state cores'. The state cores write the row's o
// back into slot k % GDN_O_IN_DEPTH of cb_o_in (k: this core's row count), which compute RMS-normalizes one row later
// (after its next row's prep); this writer then stores the output rows.
template <uint32_t Kt, uint32_t Vt, uint32_t Hv, typename Accessor>
void prep_writer(
    const Accessor& out_acc, uint32_t first_unit, uint32_t T, uint32_t tr_offset, uint32_t tr_stride, uint32_t arg) {
    constexpr uint32_t Vb = Vt / PIECES;
    constexpr uint32_t GROUP = RATIO * PIECES;
    constexpr uint32_t T_BYTES = STATE_TILE_BYTES;
    constexpr uint32_t ROW_PAGES = RATIO * Vt + 3 * Kt + RATIO * GDN_HEAD_PACK_PAGES;
    uint32_t cons_x[GROUP];
    uint32_t cons_y[GROUP];
    for (uint32_t c = 0; c < GROUP; ++c) {
        cons_x[c] = get_arg_val<uint32_t>(arg++);
        cons_y[c] = get_arg_val<uint32_t>(arg++);
    }
    // Multicast rectangles: the group's, then each head's pieces' (host _mcast_rects).
    const uint32_t group_rects = arg;
    uint32_t head_rects[RATIO];
    arg += 1 + 5 * get_arg_val<uint32_t>(arg);
    for (uint32_t j = 0; j < RATIO; ++j) {
        head_rects[j] = arg;
        arg += 1 + 5 * get_arg_val<uint32_t>(arg);
    }
    const uint32_t v_base = get_write_ptr(cb_v);
    const uint32_t kq_base = get_write_ptr(cb_kq_rows);
    const uint32_t knT_base = get_write_ptr(cb_knT);
    const uint32_t vec_base = get_write_ptr(cb_chunk_vec);
    const uint32_t mat_base = get_write_ptr(cb_chunk_mat);
    constexpr uint32_t VEC_PAGES = GDN_STATE_VEC_PAGES;
    constexpr uint32_t MAT_PAGES = GDN_HEAD_PACK_PAGES - GDN_STATE_VEC_PAGES;

    Noc noc;
    Semaphore<> granted(GDN_LANDED_FIRST_SEM_ID + INBOX_SLOTS);
    Semaphore<> o_landed(GDN_LANDED_FIRST_SEM_ID + INBOX_SLOTS + 1);
    constexpr uint32_t O_PAGES = RATIO * Vt;  // o of one tile row: the r heads' V columns, head-major
    constexpr uint32_t out_width_tiles = Hv * Vt;
    const uint32_t pair0 = first_unit / PIECES;
    const uint32_t head0 = pair0 % Hv;
    const uint32_t seq_lo = (pair0 / Hv) * T;
    const uint32_t seq_hi = seq_lo + T;
    const uint32_t first_tr = seq_lo / TILE_ROWS;
    const uint32_t last_tr = (seq_hi - 1) / TILE_ROWS;
    auto row_lo = [&](uint32_t tr) { return seq_lo > tr * TILE_ROWS ? seq_lo : tr * TILE_ROWS; };
    auto row_hi = [&](uint32_t tr) { return seq_hi < (tr + 1) * TILE_ROWS ? seq_hi : (tr + 1) * TILE_ROWS; };
    auto store = [&](uint32_t tr) {
        store_rows(out_acc, O_PAGES, tr * out_width_tiles + head0 * Vt, tr * TILE_ROWS, row_lo(tr), row_hi(tr));
    };
    uint32_t sent = 0;
    for (uint32_t tr = first_tr + tr_offset; tr <= last_tr; tr += tr_stride, ++sent) {
        const uint32_t base_row = tr * TILE_ROWS;
        const uint32_t lo = row_lo(tr);
        const uint32_t hi = row_hi(tr);
        cb_wait_front(cb_gates, 2);
        volatile tt_l1_ptr uint32_t* gates = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_gates));
        for (uint32_t j = 0; j < RATIO; ++j) {
            push_chunk_coefs(gates, (head0 + j) % TILE_ROWS, base_row, lo, hi);
        }
        cb_pop_front(cb_gates, 2);

        // Every state core of the group grants each of its rows once, in order; this core's next row is granted
        // only after this one has landed (stride >= inbox slots), so the cumulative count identifies the row.
        granted.wait_min((sent + 1) * GROUP);
        // The row's o slot is free (compute has normalized the row that used it), so the row may be sent.
        cb_reserve_back(cb_o_in, O_PAGES);
        const uint32_t slot = (tr - first_tr) % INBOX_SLOTS;
        cb_wait_front(cb_prep_send, ROW_PAGES);
        const uint32_t src = get_read_ptr(cb_prep_send);
        const uint32_t kq_src = src + RATIO * Vt * T_BYTES;
        // kn, qn (cb_kq_rows) and kn^T (cb_knT) are contiguous in the send block: two multicasts per row run.
        mcast_rects(group_rects, kq_src, kq_base + slot * 2 * Kt * T_BYTES, 2 * Kt * T_BYTES);
        mcast_rects(group_rects, kq_src + 2 * Kt * T_BYTES, knT_base + slot * Kt * T_BYTES, Kt * T_BYTES);
        for (uint32_t j = 0; j < RATIO; ++j) {
            const uint32_t head_src = kq_src + (3 * Kt + j * GDN_HEAD_PACK_PAGES) * T_BYTES;
            mcast_rects(head_rects[j], head_src, vec_base + slot * VEC_PAGES * T_BYTES, VEC_PAGES * T_BYTES);
            mcast_rects(
                head_rects[j],
                head_src + VEC_PAGES * T_BYTES,
                mat_base + slot * MAT_PAGES * T_BYTES,
                MAT_PAGES * T_BYTES);
        }
        for (uint32_t c = 0; c < GROUP; ++c) {
            const uint32_t j = c / PIECES;
            const uint32_t p = c % PIECES;
            noc_async_write(
                src + (j * Vt + p * Vb) * T_BYTES,
                get_noc_addr(cons_x[c], cons_y[c], v_base + slot * Vb * T_BYTES),
                Vb * T_BYTES);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_prep_send, ROW_PAGES);
        for (uint32_t c = 0; c < GROUP; ++c) {
            Semaphore<> landed(GDN_LANDED_FIRST_SEM_ID + slot);
            landed.up(noc, cons_x[c], cons_y[c], 1);
        }
        // Every state core writes its o of the row once; a row is sent only after the previous row's o has
        // landed everywhere, so the cumulative count identifies the row.
        o_landed.wait_min((sent + 1) * GROUP);
        cb_push_back(cb_o_in, O_PAGES);
        if (sent > 0) {
            store(tr - tr_stride);
        }
    }
    if (sent > 0) {
        store(first_tr + tr_offset + (sent - 1) * tr_stride);
    }
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
    // One call site per pipe face keeps the writer's text small (it shares the kernel-config budget).
    const bool q_injector = qk_role == GDN_QK_Q_INJECTOR;
    const bool k_injector = qk_role == GDN_QK_K_INJECTOR;
    if (q_injector || k_injector) {
        cb_wait_front(cb_qk_stage, Kt);
        const uint32_t stage = get_read_ptr(cb_qk_stage);
        if (q_injector) {
            q_channel.optional_sender(noc)->send(stage, qn_dst, part_bytes);
        }
        if (k_injector) {
            k_channel.optional_sender(noc)->send(stage, kn_dst, part_bytes);
        }
        cb_pop_front(cb_qk_stage, Kt);
    }
    if (!q_injector) {
        q_channel.optional_receiver(noc)->receive();
    }
    if (!k_injector) {
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

    constexpr auto out_args = TensorAccessorArgs<11>();
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
    const uint32_t tr_offset = get_arg_val<uint32_t>(arg++);        // tile rows handled: offset, + stride, ...
    const uint32_t tr_stride = get_arg_val<uint32_t>(arg++);
    // R-VSPLIT: NoC (x, y) of the pair's piece cores in piece order (this core included). The host runs
    // one unit per core then, so every piece of a pair walks the same tile rows in lockstep.
    uint32_t peer_x[PIECES];
    uint32_t peer_y[PIECES];
    for (uint32_t s = 0; s < PIECES && PIECES > 1 && !PREP; ++s) {
        peer_x[s] = get_arg_val<uint32_t>(arg++);
        peer_y[s] = get_arg_val<uint32_t>(arg++);
    }
    uint32_t rows_combined = 0;
    // R-PREP state core: NoC (x, y) of its group's prep cores; tile row i's o goes back to prep core i % producers.
    constexpr uint32_t MAX_PRODUCERS = STATE ? 64 : 1;
    uint32_t producers = 0;
    uint32_t producer_x[MAX_PRODUCERS];
    uint32_t producer_y[MAX_PRODUCERS];
    if constexpr (STATE) {
        producers = get_arg_val<uint32_t>(arg++);
        for (uint32_t q = 0; q < producers; ++q) {
            producer_x[q] = get_arg_val<uint32_t>(arg++);
            producer_y[q] = get_arg_val<uint32_t>(arg++);
        }
    }

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

    if constexpr (PREP) {
        prep_writer<Kt, Vt, Hv>(out_acc, first_unit, T, tr_offset, tr_stride, arg);
        return;
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

        for (uint32_t tr = first_tr + tr_offset; tr <= last_tr; tr += tr_stride) {
            const uint32_t base_row = tr * TILE_ROWS;
            const uint32_t lo = seq_lo > base_row ? seq_lo : base_row;
            const uint32_t hi = seq_hi < base_row + TILE_ROWS ? seq_hi : base_row + TILE_ROWS;

            // ---- per-token coefficient expansion (R-CHUNK: per-tile-row g / beta columns) ----
            if constexpr (!STATE) {
                cb_wait_front(cb_gates, 2);
                volatile tt_l1_ptr uint32_t* gates =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_gates));
                if constexpr (CHUNKED) {
                    push_chunk_coefs(gates, gate_col, base_row, lo, hi);
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
            }

            if (!CHUNKED && qk_role != GDN_QK_OWN) {
                land_qk_rows(q_channel, k_channel, qk_role, Kt);
            }

            // ---- partial sum o^2 all-gather (R-VSPLIT) ----
            // Each piece writes its partial into slot `piece` of the gather buffer on every piece core
            // (same L1 address everywhere: identical CB layout, identical tile-row count), then bumps
            // the peers' counters. cb_osum_gather holds two tile rows: a peer can be at most one tile
            // row ahead, because it cannot finish row i+1 without this core's partial of row i+1, which
            // compute produces only after it has consumed row i's slots.
            if constexpr (STATE) {
                // ---- R-PREP: o of the piece into its place in slot k % GDN_O_IN_DEPTH of the row's prep core ----
                const uint32_t i = tr - first_tr;
                const uint32_t q = i % producers;
                const uint32_t head_in_group = head % RATIO;
                const uint32_t o_slot = (i / producers) % GDN_O_IN_DEPTH;
                const uint32_t dst =
                    get_write_ptr(cb_o_in) + (o_slot * RATIO * Vt + head_in_group * Vt + v_first) * STATE_TILE_BYTES;
                cb_wait_front(cb_o_acc, Vb);
                noc_async_write(
                    get_read_ptr(cb_o_acc), get_noc_addr(producer_x[q], producer_y[q], dst), Vb * STATE_TILE_BYTES);
                noc_async_write_barrier();
                cb_pop_front(cb_o_acc, Vb);
                Noc noc;
                Semaphore<> o_landed(GDN_LANDED_FIRST_SEM_ID + INBOX_SLOTS + 1);
                o_landed.up(noc, producer_x[q], producer_y[q], 1);
                continue;
            }
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

            store_rows(out_acc, Vb, tr * out_width_tiles + head * Vt + v_first, base_row, lo, hi);
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
