// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gated_delta_net writer (BRISC).
//
// Per unit (pair, piece), per tile row: expands the pair's (alpha, beta) gate elements into three
// column-0 coefficient pages per token (alpha at row p; beta at row p; 1 at row p); with
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
#include "gated_delta_net_qk_rows.hpp"

namespace {

// CB indices and GDN_* spec constants are host defines (gated_delta_net_program_descriptor.py).
constexpr uint32_t cb_state_out = GDN_CB_STATE_OUT;
constexpr uint32_t cb_gates = GDN_CB_GATES;
constexpr uint32_t cb_tok_coef = GDN_CB_TOK_COEF;
constexpr uint32_t cb_out_rows = GDN_CB_OUT_ROWS;
constexpr uint32_t cb_window = GDN_CB_WINDOW;
constexpr uint32_t cb_window_zeroed = GDN_CB_WINDOW_ZEROED;
constexpr uint32_t cb_hist_stage = GDN_CB_HIST_STAGE;
constexpr uint32_t cb_scratch = GDN_CB_SCRATCH;
constexpr uint32_t cb_osum_part = GDN_CB_OSUM_PART;
constexpr uint32_t cb_osum_gather = GDN_CB_OSUM_GATHER;
constexpr uint32_t cb_kq_rows = GDN_CB_KQ_ROWS;
constexpr uint32_t cb_kq_rows_f32 = GDN_CB_KQ_ROWS_F32;
constexpr uint32_t cb_qk_stage = GDN_CB_QK_STAGE;
constexpr uint32_t cb_prep_send = GDN_CB_PREP_SEND;
constexpr uint32_t cb_in_kq = GDN_CB_IN_KQ;
constexpr uint32_t cb_in_knt = GDN_CB_IN_KNT;
constexpr uint32_t cb_in_mat = GDN_CB_IN_MAT;
constexpr uint32_t cb_in_vec = GDN_CB_IN_VEC;
constexpr uint32_t cb_in_v = GDN_CB_IN_V;
constexpr uint32_t cb_o_acc = GDN_CB_O_ACC;
constexpr uint32_t cb_o_in = GDN_CB_O_IN;
constexpr uint32_t cb_tri_mask = GDN_CB_TRI_MASK;

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
constexpr uint32_t RATIO = get_compile_time_arg_val(10);               // R-PREP: value heads per prep group
constexpr bool PREP = ROLE == GDN_ROLE_PREP;
constexpr bool STATE = ROLE == GDN_ROLE_STATE;
constexpr bool QK_CORE = ROLE == GDN_ROLE_QK;  // R-QKCORE: multicasts its key head's qn or kn, nothing else
constexpr bool PREP_QK = ROLE == GDN_ROLE_PREP_QK;  // R-PREP q / k core: writes qn or kn to its prep cores (RATIO)
constexpr bool HOLD_QK = GDN_HOLD_QK != 0;     // compute holds the landed q/k rows for every unit (see compute)
constexpr bool UNIT_LAG = GDN_UNIT_LAG != 0;   // compute normalizes unit u after unit u + 1's token step
constexpr uint32_t STATE_STAGGER = GDN_STATE_STAGGER;  // compute's rotation of the final-state tiles (see compute)
constexpr bool STATE_SPLIT = GDN_STATE_SPLIT != 0;     // the reader writes this core's reader_share of the state tiles
constexpr uint32_t cb_state_done = GDN_CB_STATE_DONE;  // STATE_SPLIT: the reader's tiles of a state have left
constexpr uint32_t TILE_WORDS = 1024;  // cb_gates / cb_tok_coef are Float32 compute buffers
constexpr uint32_t ONE_BITS = 0x3f800000u;  // 1.0f

// R-PREP: base of this core's CB region, from one of its role's remotely written buffers `cb` at byte offset `offset`
// from it (host STATE_REMOTE_CBS / PREP_REMOTE_CBS; a q / k core's cb_qk_stage at 0). A buffer the other role owns is
// at that base plus its GDN_OFFSET_* on every core of that role.
FORCE_INLINE uint32_t cb_region_base(uint32_t cb, uint32_t offset) {
    const auto& cb_if = get_local_cb_interface(cb);
    return cb_if.fifo_limit - cb_if.fifo_size - offset;
}

FORCE_INLINE uint32_t row_offset(uint32_t p) { return (p >> 4) * 2 * FACE_BYTES + (p & 15) * SEG_BYTES; }

// Output face-row segments are written directly: Tensix-to-DRAM writes need only L1 alignment.
// Word index of element (r, c) of an fp32 tile (four 16x16 faces).
FORCE_INLINE uint32_t elem_index(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

// store_rows_block: n output tiles from cb_out_rows to pages first_page.. of tile row [base_row, +32), whole tiles
// when [lo, hi) is the whole tile row, else face-row segments of those rows only.
// RETIRE = false leaves the barrier and the pop to the caller.
template <bool RETIRE = true, typename Accessor>
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
    if constexpr (RETIRE) {
        noc_async_write_barrier();
        cb_pop_front(cb_out_rows, n);
    }
}

// One multicast write per rectangle of the runtime-arg list at `arg` (count, then x_lo, y_lo, x_hi, y_hi, cores each),
// a unicast write for a one-core rectangle; the sender lies outside every rectangle. NOC 1 takes the corners in the
// opposite order.
FORCE_INLINE void mcast_rects(uint32_t arg, uint32_t src, uint32_t dst, uint32_t bytes) {
    const uint32_t rects = get_arg_val<uint32_t>(arg++);
    for (uint32_t r = 0; r < rects; ++r, arg += 5) {
        const uint32_t x_lo = get_arg_val<uint32_t>(arg), y_lo = get_arg_val<uint32_t>(arg + 1);
        const uint32_t x_hi = get_arg_val<uint32_t>(arg + 2), y_hi = get_arg_val<uint32_t>(arg + 3);
        const uint32_t cores = get_arg_val<uint32_t>(arg + 4);
        if (cores == 1) {
            noc_async_write(src, get_noc_addr(x_lo, y_lo, dst), bytes);
            continue;
        }
        const uint64_t dst_noc = noc_index == 0 ? get_noc_multicast_addr(x_lo, y_lo, x_hi, y_hi, dst)
                                                : get_noc_multicast_addr(x_hi, y_hi, x_lo, y_lo, dst);
        noc_async_write_multicast(src, dst_noc, bytes, cores);
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
// prepped tiles arrive in cb_prep_send as one block per value head j, [exp(c) | QD, M | kq (2 Kt) | knT (Kt) | v (Vt)]
// (descriptor IN_MAT_PAGES), part by part, and go to the head's state cores (unit order: head j, piece p) at inbox slot
// i % INBOX_SLOTS of cb_in_kq, cb_in_knt, cb_in_mat, cb_in_vec (multicast to the head's pieces) and cb_in_v (the
// piece's Vb columns). Those buffers lead the state cores' CB region, cb_o_in leads this core's, so their addresses
// are this core's region base plus their offsets. The state cores write the row's o
// back into slot k % GDN_O_IN_DEPTH of cb_o_in (k: this core's row count), which compute RMS-normalizes one row later
// (after its next row's prep); this writer then stores the output rows.
template <uint32_t Kt, uint32_t Vt, uint32_t Hv, typename Accessor>
void prep_writer(
    const Accessor& out_acc, uint32_t first_unit, uint32_t T, uint32_t tr_offset, uint32_t tr_stride, uint32_t arg) {
    constexpr uint32_t Vb = Vt / PIECES;
    constexpr uint32_t GROUP = RATIO * PIECES;
    constexpr uint32_t T_BYTES = STATE_TILE_BYTES;
    constexpr uint32_t VEC_PAGES = GDN_IN_VEC_PAGES;
    constexpr uint32_t HEAD_PAGES = VEC_PAGES + GDN_IN_MAT_PAGES + 3 * Kt + Vt;
    uint32_t cons_x[GROUP];
    uint32_t cons_y[GROUP];
    for (uint32_t c = 0; c < GROUP; ++c) {
        cons_x[c] = get_arg_val<uint32_t>(arg++);
        cons_y[c] = get_arg_val<uint32_t>(arg++);
    }
    // Multicast rectangles of each head's pieces (host _mcast_rects).
    uint32_t head_rects[RATIO];
    for (uint32_t j = 0; j < RATIO; ++j) {
        head_rects[j] = arg;
        arg += 1 + 5 * get_arg_val<uint32_t>(arg);
    }
    const uint32_t region = cb_region_base(cb_o_in, GDN_OFFSET_O_IN);
    const uint32_t v_base = region + GDN_OFFSET_IN_V;
    const uint32_t kq_base = region + GDN_OFFSET_IN_KQ;
    const uint32_t knT_base = region + GDN_OFFSET_IN_KNT;
    const uint32_t mat_base = region + GDN_OFFSET_IN_MAT;
    const uint32_t vec_base = region + GDN_OFFSET_IN_VEC;

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
    // Row k's o lands (and its output is stored) during later rows, so that nothing ties the next row's preparation
    // to the state cores' progress: iteration k expands row k's coefficients, lands row k - 1's o, stores row k - 2,
    // then sends row k once it is granted.
    auto land_o = [&](uint32_t row_count) {
        // Every state core writes its o of a row once; a row is sent only after the previous row's o has landed
        // everywhere, so the cumulative count identifies the row.
        o_landed.wait_min(row_count * GROUP);
        cb_push_back(cb_o_in, O_PAGES);
    };
    const uint32_t first_row = first_tr + tr_offset;
    uint32_t sent = 0;
    for (uint32_t tr = first_row; tr <= last_tr; tr += tr_stride, ++sent) {
        GDN_ZONE("gdn_wr_prep_row");
        const uint32_t base_row = tr * TILE_ROWS;
        const uint32_t lo = row_lo(tr);
        const uint32_t hi = row_hi(tr);
        cb_wait_front(cb_gates, 2);
        volatile tt_l1_ptr uint32_t* gates = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_gates));
        for (uint32_t j = 0; j < RATIO; ++j) {
            push_chunk_coefs(gates, (head0 + j) % TILE_ROWS, base_row, lo, hi);
        }
        cb_pop_front(cb_gates, 2);
        if (sent >= 1) {
            land_o(sent);
        }
        if (sent >= 2) {
            store(tr - 2 * tr_stride);
        }

        // Every state core of the group grants each of its rows once, in order; this core's next row is granted
        // only after this one has landed (stride >= inbox slots), so the cumulative count identifies the row.
        granted.wait_min((sent + 1) * GROUP);
        // The row's o slot is free (compute has normalized the row that used it), so the row may be sent.
        cb_reserve_back(cb_o_in, O_PAGES);
        const uint32_t slot = (tr - first_tr) % INBOX_SLOTS;
        // One head block at a time, each part sent as compute pushes it (pack order: exp(c) | QD, M | kq | knT | v).
        for (uint32_t j = 0; j < RATIO; ++j) {
            cb_wait_front(cb_prep_send, VEC_PAGES + GDN_IN_MAT_PAGES);
            const uint32_t vec_src = get_read_ptr(cb_prep_send);
            const uint32_t mat_src = vec_src + VEC_PAGES * T_BYTES;
            const uint32_t kq_src = mat_src + GDN_IN_MAT_PAGES * T_BYTES;
            const uint32_t knT_src = kq_src + 2 * Kt * T_BYTES;
            const uint32_t v_src = knT_src + Kt * T_BYTES;
            mcast_rects(head_rects[j], vec_src, vec_base + slot * VEC_PAGES * T_BYTES, VEC_PAGES * T_BYTES);
            mcast_rects(
                head_rects[j], mat_src, mat_base + slot * GDN_IN_MAT_PAGES * T_BYTES, GDN_IN_MAT_PAGES * T_BYTES);
            cb_wait_front(cb_prep_send, VEC_PAGES + GDN_IN_MAT_PAGES + 2 * Kt);
            mcast_rects(head_rects[j], kq_src, kq_base + slot * 2 * Kt * T_BYTES, 2 * Kt * T_BYTES);
            cb_wait_front(cb_prep_send, HEAD_PAGES - Vt);
            mcast_rects(head_rects[j], knT_src, knT_base + slot * Kt * T_BYTES, Kt * T_BYTES);
            cb_wait_front(cb_prep_send, HEAD_PAGES);
            for (uint32_t p = 0; p < PIECES; ++p) {
                const uint32_t c = j * PIECES + p;
                noc_async_write(
                    v_src + p * Vb * T_BYTES,
                    get_noc_addr(cons_x[c], cons_y[c], v_base + slot * Vb * T_BYTES),
                    Vb * T_BYTES);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_prep_send, HEAD_PAGES);
        }
        for (uint32_t c = 0; c < GROUP; ++c) {
            Semaphore<> landed(GDN_LANDED_FIRST_SEM_ID + slot);
            landed.up(noc, cons_x[c], cons_y[c], 1);
        }
    }
    if (sent >= 1) {
        land_o(sent);
    }
    for (uint32_t k = sent >= 2 ? sent - 2 : 0; k < sent; ++k) {
        store(first_row + k * tr_stride);
    }
}

// R-PREP q / k core (PREP_SHARE_QK). Unit = its key head's first group; tile rows offset, offset + stride, ... of the
// sequence. Each row's qn (kn) from cb_qk_stage goes to tiles [Kt, 2Kt) ([0, Kt)) of slot k % GDN_PREP_KQ_DEPTH of
// cb_kq_rows on the RATIO prep cores of its stride position (k: this core's row count), once each of them has granted
// that row; then their qn (kn) landed counters are bumped. cb_kq_rows sits at the same offset from the CB region base
// on every prep core, cb_qk_stage leads this core's region.
template <uint32_t Kt, uint32_t Hv>
void prep_qk_writer(
    uint32_t first_unit, uint32_t T, uint32_t tr_offset, uint32_t tr_stride, uint32_t qk_role, uint32_t arg) {
    uint32_t dst_x[RATIO];
    uint32_t dst_y[RATIO];
    for (uint32_t d = 0; d < RATIO; ++d) {
        dst_x[d] = get_arg_val<uint32_t>(arg++);
        dst_y[d] = get_arg_val<uint32_t>(arg++);
    }
    constexpr uint32_t part_bytes = Kt * STATE_TILE_BYTES;
    constexpr uint32_t landed_first = GDN_LANDED_FIRST_SEM_ID + INBOX_SLOTS + 2;  // qn, then kn (see the reader)
    const bool is_q = qk_role == GDN_QK_Q_INJECTOR;
    const uint32_t dst_base = cb_region_base(cb_qk_stage, 0) + GDN_OFFSET_KQ_ROWS + (is_q ? part_bytes : 0);
    Noc noc;
    Semaphore<> landed(landed_first + (is_q ? 0 : 1));
    const uint32_t seq_lo = (first_unit / PIECES / Hv) * T;
    const uint32_t first_tr = seq_lo / TILE_ROWS;
    const uint32_t last_tr = (seq_lo + T - 1) / TILE_ROWS;
    uint32_t sent = 0;
    for (uint32_t tr = first_tr + tr_offset; tr <= last_tr; tr += tr_stride, ++sent) {
        GDN_ZONE("gdn_wr_qk_row");
        cb_wait_front(cb_qk_stage, Kt);
        const uint32_t src = get_read_ptr(cb_qk_stage);
        const uint32_t dst = dst_base + (sent % GDN_PREP_KQ_DEPTH) * 2 * part_bytes;
        for (uint32_t d = 0; d < RATIO; ++d) {
            Semaphore<>(landed_first + 2 + d).wait_min(sent + 1);
            noc_async_write(src, get_noc_addr(dst_x[d], dst_y[d], dst), part_bytes);
        }
        noc_async_write_barrier();
        for (uint32_t d = 0; d < RATIO; ++d) {
            landed.up(noc, dst_x[d], dst_y[d], 1);
        }
        cb_pop_front(cb_qk_stage, Kt);
    }
}

// Lands kn (cb_kq_rows tiles [0, Kt)) and qn ([Kt, 2Kt)) of one tile row. An injector multicasts its staged
// part (qn from the q injector, kn from the k injector), its own rows included (the sender lies inside its
// group, so a send from another address loops back and is complete when send() returns); every core receives
// the parts it did not prep. Every core has the same CB layout and R-BCAST hands off one block per core, so a
// receiver's write pointer is the address the injector targets.
// R-QKCORE pair cores receive once per call (`receive` on the first unit only): a later unit re-pushes the same
// rows, which compute's pop left in place (cb_kq_rows holds exactly one block, so the write pointer wraps onto them).
template <typename QChannel, typename KChannel>
void land_qk_rows(const QChannel& q_channel, const KChannel& k_channel, uint32_t qk_role, uint32_t Kt, bool receive) {
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
    if (!q_injector && receive) {
        q_channel.optional_receiver(noc)->receive();
    }
    if (!k_injector && receive) {
        k_channel.optional_receiver(noc)->receive();
    }
    cb_push_back(cb_kq_rows, 2 * Kt);
    cb_push_back(cb_kq_rows_f32, 2 * Kt);
}

}  // namespace

// Issues the piece's Kt x Vb state tiles of cb_state_out to recurrent_state_out (pair state page `first_page` on),
// except the reader's (reader_share); the caller barriers and calls pop_state.
template <uint32_t Kt, uint32_t Vt, uint32_t Vb, typename Accessor>
void store_state(
    const Accessor& state_out, uint32_t first_page, uint32_t unit_idx, uint32_t reader_units, uint32_t reader_share) {
    // Compute pushes the final state in update batches: each tile goes out as soon as it is packed.
    const uint32_t state_base = get_read_ptr(cb_state_out);
    for (uint32_t j = 0; j < Kt * Vb; ++j) {
        if (gdn::state_tile_on_reader(unit_idx, j, reader_units, reader_share)) {
            continue;
        }
        cb_wait_front(cb_state_out, j + 1);
        const uint32_t page = gdn::state_out_page<Kt, Vt, Vb>(first_page, j, STATE_STAGGER);
        noc_async_write(state_base + j * STATE_TILE_BYTES, state_out.get_noc_addr(page), STATE_TILE_BYTES);
    }
}

// Frees the front state of cb_state_out (its writes have completed), once the reader's have too (reader_writes).
FORCE_INLINE void pop_state(uint32_t n_state, bool reader_writes) {
    if (STATE_SPLIT && reader_writes) {
        cb_wait_front(cb_state_done, 1);
        cb_pop_front(cb_state_done, 1);
    }
    cb_pop_front(cb_state_out, n_state);
}

void kernel_main() {
    const gdn::NocDrainOnReturn drain;
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

    // Buffer addresses are common runtime args (see the reader).
    const auto out_acc = TensorAccessor(out_args, get_common_arg_val<uint32_t>(0), OUT_TILE_BYTES);
    const auto state_out = TensorAccessor(state_out_args, get_common_arg_val<uint32_t>(1), STATE_TILE_BYTES);
    uint32_t arg = 0;
    const uint32_t first_unit = get_arg_val<uint32_t>(arg++);
    const uint32_t num_units = get_arg_val<uint32_t>(arg++);
    const uint32_t T = get_arg_val<uint32_t>(arg++);
    // Compile-time when every core of the kernel group shares it (descriptor GDN_GROUP_QK_ROLE).
    const uint32_t qk_role =
        GDN_GROUP_QK_ROLE == GDN_QK_PER_CORE ? get_arg_val<uint32_t>(arg) : uint32_t{GDN_GROUP_QK_ROLE};
    ++arg;
    const uint32_t zero_slots = get_arg_val<uint32_t>(arg++);       // cb_window slots this core's reader uses
    const uint32_t zero_slot_bytes = get_arg_val<uint32_t>(arg++);  // one window group
    const uint32_t tr_offset = get_arg_val<uint32_t>(arg++);        // tile rows handled: offset, + stride, ...
    const uint32_t tr_stride = get_arg_val<uint32_t>(arg++);
    const uint32_t unit_stride = get_arg_val<uint32_t>(arg++);  // units first_unit, + stride, ... (R-QKCORE: Hv)
    // R-VSPLIT: NoC (x, y) of the pair's piece cores in piece order (this core included). The host runs
    // one unit per core then, so every piece of a pair walks the same tile rows in lockstep.
    uint32_t peer_x[PIECES];
    uint32_t peer_y[PIECES];
    for (uint32_t s = 0; s < PIECES && PIECES > 1 && !PREP && !PREP_QK; ++s) {
        peer_x[s] = get_arg_val<uint32_t>(arg++);
        peer_y[s] = get_arg_val<uint32_t>(arg++);
    }
    uint32_t rows_combined = 0;
    // R-PREP state core: NoC (x, y) of its group's prep cores; tile row i's o goes back to prep core i % producers.
    constexpr uint32_t MAX_PRODUCERS = STATE ? GDN_MAX_PREP_CORES_PER_GROUP : 1;
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
    // STATE_SPLIT: the reader's final-state tiles (gdn::state_tile_on_reader).
    const bool split_args = STATE_SPLIT && !QK_CORE && !PREP && !PREP_QK && !STATE;
    const uint32_t reader_units = split_args ? get_arg_val<uint32_t>(arg++) : 0;
    const uint32_t reader_share = split_args ? get_arg_val<uint32_t>(arg++) : 0;
    const bool reader_writes = reader_units != 0 || reader_share != 0;

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

    if constexpr (CHUNKED && !STATE && !PREP_QK) {
        // Lower-triangular 0/1 mask (column <= row) of the chunk decay matrix, then the identity, once per kernel:
        // zeroed over the NoC, then only the ones are stored. Built here, after the window zeroing: the reader's first
        // DRAM reads are on the critical path, this RISC is idle until the first gates.
        Noc noc;
        CircularBuffer mask_buf(cb_tri_mask);
        cb_reserve_back(cb_tri_mask, 2);
        noc.async_write_zeros(mask_buf, 2 * STATE_TILE_BYTES, {.offset_bytes = 0});
        noc.write_zeros_l1_barrier();
        volatile tt_l1_ptr uint32_t* mask = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_tri_mask));
        for (uint32_t r = 0; r < TILE_ROWS; ++r) {
            const uint32_t row = (r >> 4) * 2 * 256 + (r & 15) * 16;
            for (uint32_t c = 0; c <= r; ++c) {
                mask[row + (c >> 4) * 256 + (c & 15)] = ONE_BITS;
            }
            mask[TILE_WORDS + row + (r >> 4) * 256 + (r & 15)] = ONE_BITS;  // identity (R-PREP inverse)
        }
        cb_push_back(cb_tri_mask, 2);
    }

    if constexpr (PREP) {
        prep_writer<Kt, Vt, Hv>(out_acc, first_unit, T, tr_offset, tr_stride, arg);
        return;
    }
    if constexpr (PREP_QK) {
        prep_qk_writer<Kt, Hv>(first_unit, T, tr_offset, tr_stride, qk_role, arg);
        return;
    }
    if constexpr (QK_CORE) {
        // Window: per column, zero its slot, copy the history rows from history_split(B) on, hand it to the reader.
        GDN_ZONE("gdn_wr_qk");
        Noc noc;
        CircularBuffer window_buf(cb_window);
        const uint32_t batch = num_units;
        const uint32_t cs_tiles = (GDN_HISTORY * batch + TILE_ROWS - 1) / TILE_ROWS;
        const uint32_t cs_stage = get_write_ptr(cb_hist_stage) + Kt * OUT_TILE_BYTES;
        const uint32_t win = get_write_ptr(cb_window);
        constexpr uint32_t slot_bytes = GDN_CONV_TAPS * OUT_TILE_BYTES;
        cb_wait_front(cb_scratch, 1);  // staging landed
        for (uint32_t col = 0; col < Kt; ++col) {
            noc.async_write_zeros(window_buf, slot_bytes, {.offset_bytes = col * slot_bytes});
            noc.write_zeros_l1_barrier();
            gdn::copy_history_rows<OUT_TILE_BYTES, GDN_HISTORY>(
                cs_stage + col * cs_tiles * OUT_TILE_BYTES, win + col * slot_bytes, gdn::history_split(batch), batch);
            noc_async_read_barrier();
            cb_reserve_back(cb_window_zeroed, 1);
            cb_push_back(cb_window_zeroed, 1);
        }
        cb_pop_front(cb_scratch, 1);
        // qn lands in cb_kq_rows tiles [Kt, 2Kt) of the key head's pair cores, kn in [0, Kt): their untouched base.
        const uint32_t part_bytes = Kt * STATE_TILE_BYTES;
        const uint32_t kn_dst = get_write_ptr(cb_kq_rows);
        cb_wait_front(cb_qk_stage, Kt);
        if (qk_role == GDN_QK_Q_INJECTOR) {
            q_channel.optional_sender(noc)->send(get_read_ptr(cb_qk_stage), kn_dst + part_bytes, part_bytes);
        } else {
            k_channel.optional_sender(noc)->send(get_read_ptr(cb_qk_stage), kn_dst, part_bytes);
        }
        cb_pop_front(cb_qk_stage, Kt);
        return;
    }

    // A unit's last output rows and its state are written without waiting: the barrier and the pops wait until the
    // next unit's store point, after its coefficients and q/k rows are handed to compute. A non-last unit's state is
    // also issued only there (issuing blocks this RISC while the NoC is congested; cb_state_out holds two states).
    bool retiring = false;                 // output rows (and an issued state) await the barrier and their pops
    bool state_issued = false;             // the state at the front of cb_state_out has been issued
    uint32_t pending_state = 0xffffffffu;  // first state page of an unissued finished state
    uint32_t pending_unit = 0;             // its unit index on this core
    auto retire = [&]() {
        if (pending_state != 0xffffffffu) {
            store_state<Kt, Vt, Vb>(state_out, pending_state, pending_unit, reader_units, reader_share);
            pending_state = 0xffffffffu;
            state_issued = true;
        }
        if (retiring) {
            noc_async_write_barrier();
            cb_pop_front(cb_out_rows, Vb);
            retiring = false;
        }
        if (state_issued) {
            noc_async_write_barrier();
            pop_state(n_state, reader_writes);
            state_issued = false;
        }
    };
    const bool hold_qk = HOLD_QK && qk_role == GDN_QK_RECEIVER && unit_stride == Hv * PIECES && T == 1;
    const bool lag = UNIT_LAG && hold_qk && num_units > 1;
    // UNIT_LAG: the output rows of the previous unit, stored once compute has normalized them (after this unit's step).
    uint32_t lag_page = 0, lag_base = 0, lag_lo = 0, lag_hi = 0;
    for (uint32_t unit_idx = 0; unit_idx < num_units; ++unit_idx) {
        const uint32_t unit = first_unit + unit_idx * unit_stride;
        GDN_ZONE("gdn_wr_unit");
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
                        coef[e] = at_p ? alpha : 0u;
                        coef[TILE_WORDS + e] = at_p ? beta : 0u;
                        coef[2 * TILE_WORDS + e] = at_p ? ONE_BITS : 0u;  // row-p selector
                    }
                    cb_push_back(cb_tok_coef, GDN_TOK_COEF_PAGES);
                }
                cb_pop_front(cb_gates, 2);
            }

            if (!CHUNKED && qk_role != GDN_QK_OWN && !(hold_qk && unit_idx > 0)) {
                land_qk_rows(q_channel, k_channel, qk_role, Kt, unit_idx == 0);
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
                const uint32_t dst = cb_region_base(cb_in_kq, GDN_OFFSET_IN_KQ) + GDN_OFFSET_O_IN +
                                     (o_slot * RATIO * Vt + head_in_group * Vt + v_first) * STATE_TILE_BYTES;
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

            // The state is final after the last tile row's update, before compute's RMS norm produces the output
            // rows: its writes go out first and overlap that norm.
            retire();
            if (lag) {
                // T = 1: the unit's only tile row. The last unit's state goes out before the previous unit's rows,
                // which compute produces only after that state's update.
                if (unit_idx + 1 == num_units) {
                    store_state<Kt, Vt, Vb>(state_out, pair * Kt * Vt + v_first, unit_idx, reader_units, reader_share);
                    state_issued = true;
                } else {
                    pending_state = pair * Kt * Vt + v_first;
                    pending_unit = unit_idx;
                }
                if (unit_idx > 0) {
                    store_rows<false>(out_acc, Vb, lag_page, lag_base, lag_lo, lag_hi);
                    retiring = true;
                }
                lag_page = tr * out_width_tiles + head * Vt + v_first;
                lag_base = base_row;
                lag_lo = lo;
                lag_hi = hi;
                continue;
            }
            if (tr == last_tr) {
                if (unit_idx + 1 == num_units) {
                    store_state<Kt, Vt, Vb>(state_out, pair * Kt * Vt + v_first, unit_idx, reader_units, reader_share);
                    state_issued = true;
                } else {
                    pending_state = pair * Kt * Vt + v_first;
                    pending_unit = unit_idx;
                }
                store_rows<false>(out_acc, Vb, tr * out_width_tiles + head * Vt + v_first, base_row, lo, hi);
                retiring = true;
            } else {
                store_rows(out_acc, Vb, tr * out_width_tiles + head * Vt + v_first, base_row, lo, hi);
            }
        }
        if constexpr (STATE) {
            store_state<Kt, Vt, Vb>(state_out, pair * Kt * Vt + v_first, unit_idx, reader_units, reader_share);
            noc_async_write_barrier();
            pop_state(n_state, reader_writes);
        }
    }
    if (lag) {
        // The previous rows' writes must have left cb_out_rows before the last unit's rows are read from its front.
        noc_async_writes_flushed();
        cb_pop_front(cb_out_rows, Vb);
        retiring = false;
        store_rows<false>(out_acc, Vb, lag_page, lag_base, lag_lo, lag_hi);
        retiring = true;
    }
    retire();
}
