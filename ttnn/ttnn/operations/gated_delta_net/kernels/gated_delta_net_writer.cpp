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
constexpr uint32_t cb_state_in = GDN_CB_STATE_IN;
constexpr uint32_t cb_state_in_mm = GDN_CB_STATE_IN_MM;
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
constexpr uint32_t cb_rinv = GDN_CB_RINV;
constexpr uint32_t cb_rinv_in = GDN_CB_RINV_IN;

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
constexpr bool NORM_SPLIT = GDN_PREP_NORM_SPLIT != 0;  // R-PREP: a row's o goes to its norm core (see prep_writer)
// Host SEQ_SPLIT (see the reader): a segment's final state goes to the next segment's state core of the piece
// (cb_seg_in behind P), the last segment's to DRAM; a P core sends its piece of P to the head's state cores of its
// segment and then, row by row on their credits, the Oh_P it held in cb_reads.
constexpr uint32_t SEGS = GDN_SEQ_SPLIT;
constexpr bool SPLIT = SEGS > 1;
constexpr uint32_t cb_seg_in = GDN_CB_SEG_IN;
constexpr uint32_t cb_seg_row = GDN_CB_SEG_ROW;
constexpr uint32_t cb_reads = GDN_CB_READS;
constexpr uint32_t SEG_SEM_ID = GDN_LANDED_FIRST_SEM_ID + INBOX_SLOTS;
// Host STATE_NORM_TAIL: the state cores normalize the last TAIL tile rows of a sequence themselves (partial sums
// gathered over the pieces, output rows stored here); the prep cores' norm units end before those rows.
constexpr uint32_t TAIL = GDN_NORM_TAIL;
constexpr bool HOLD_QK = GDN_HOLD_QK != 0;     // compute holds the landed q/k rows for every unit (see compute)
constexpr bool QK_RAW = GDN_QK_RAW != 0;       // q / k cores send y, then the inverse norms (host QK_RAW_HANDOFF)
constexpr bool UNIT_LAG = GDN_UNIT_LAG != 0;   // compute normalizes unit u after unit u + 1's token step
constexpr uint32_t STATE_STAGGER = GDN_STATE_STAGGER;  // compute's rotation of the final-state tiles (see compute)
constexpr bool STATE_SPLIT = GDN_STATE_SPLIT != 0;     // the reader writes this core's reader_share of the state tiles
constexpr uint32_t cb_state_done = GDN_CB_STATE_DONE;  // STATE_SPLIT: the reader's tiles of a state have left
constexpr bool WRITER_ASYNC = GDN_WRITER_ASYNC != 0;   // UNIT_LAG cores: no unit's coefficients wait for a write
constexpr bool ASYNC_STATE_HOLD = GDN_WRITER_ASYNC != 2;
constexpr uint32_t STATE_WINDOW = GDN_STATE_WRITE_WINDOW;  // final-state tiles in flight at most (0: no bound)
constexpr uint32_t STATE_TRID = 12, ROWS_TRID = 14;        // write transaction ids: states (12, 13), output rows
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

// The three coefficient pages of the token at row p of a tile row: alpha, beta and 1 of gate column gate_col at
// (p, 0), 0 elsewhere.
void push_token_coefs(volatile tt_l1_ptr uint32_t* gates, uint32_t p, uint32_t gate_col) {
    const uint32_t alpha = gates[elem_index(p, gate_col)];
    const uint32_t beta = gates[TILE_WORDS + elem_index(p, gate_col)];
    cb_reserve_back(cb_tok_coef, GDN_TOK_COEF_PAGES);
    volatile tt_l1_ptr uint32_t* coef = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_tok_coef));
    for (uint32_t r = 0; r < TILE_ROWS; ++r) {
        const uint32_t e = elem_index(r, 0);
        const bool at_p = (r == p);
        coef[e] = at_p ? alpha : 0u;
        coef[TILE_WORDS + e] = at_p ? beta : 0u;
        coef[2 * TILE_WORDS + e] = at_p ? ONE_BITS : 0u;  // row-p selector
    }
    // TOKEN_VECTOR_FUSED reads the selector elementwise on p's 4-row band: every column there (~1.5 us
    // of this RISC at decode B=1, off its critical path there).
    for (uint32_t r = p & ~3u; r < (p & ~3u) + 4 && GDN_TOKEN_VECTOR_FUSED; ++r) {
        for (uint32_t c = 1; c < TILE_ROWS; ++c) {
            coef[2 * TILE_WORDS + elem_index(r, c)] = r == p ? ONE_BITS : 0u;
        }
    }
    cb_push_back(cb_tok_coef, GDN_TOK_COEF_PAGES);
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
        if constexpr (GDN_PREP_M_BETA != 0) {
            // diag(beta) (PREP_M_BETA; the page is zero elsewhere, see prep_writer), up by the mean loss of the FPU's
            // truncations in quarter units of the last operand bit. Zero, denormal, inf and NaN stay as they are.
            const uint32_t b = coef[TILE_WORDS + e];
            const bool normal = (b >> 23) - 1 < 0xfd;
            coef[2 * TILE_WORDS + elem_index(p, p)] =
                normal ? b + ((((b & 0x7fffffu) | 0x800000u) * GDN_M_BETA_UNBIAS_Q) >> 12) : b;
        } else if constexpr (GDN_CHUNK_COEF_PAGES > 2) {
            coef[2 * TILE_WORDS + e] = coef[TILE_WORDS + e] ^ 0x80000000u;  // -beta (STATE_FUSED_RHS)
        }
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
// (after its next row's prep); this writer then stores the output rows. NORM_SPLIT: see norm_offset below.
template <uint32_t Kt, uint32_t Vt, uint32_t Hv, typename Accessor>
void prep_writer(
    const Accessor& out_acc, uint32_t first_unit, uint32_t T, uint32_t tr_offset, uint32_t tr_stride, uint32_t arg) {
    // NORM_SPLIT: the norm units (tile row i, head j of the group: unit i * RATIO + j) this core normalizes and
    // stores: offset, + stride, ... Its k-th unit's o lands in slot k of cb_o_in (Vt tiles; never reused), counted by
    // the semaphore LANDED_FIRST + k (those ids count inbox landings on the state cores only).
    const uint32_t norm_offset = get_arg_val<uint32_t>(arg++);
    const uint32_t norm_stride = get_arg_val<uint32_t>(arg++);
    constexpr uint32_t Vb = Vt / PIECES;
    constexpr uint32_t GROUP = RATIO * PIECES;
    constexpr uint32_t T_BYTES = STATE_TILE_BYTES;
    constexpr uint32_t VEC_PAGES = GDN_IN_VEC_PAGES;
    constexpr uint32_t HEAD_PAGES = VEC_PAGES + GDN_IN_MAT_PAGES + 3 * Kt + Vt;
    // SEQ_SPLIT: a segment 1..'s rows go to the P half's cores as well (after the group's: no v for them).
    constexpr uint32_t MAX_CONS = SPLIT ? 2 * GROUP : GROUP;
    const uint32_t n_cons = SPLIT ? get_arg_val<uint32_t>(arg++) : GROUP;
    uint32_t cons_x[MAX_CONS];
    uint32_t cons_y[MAX_CONS];
    for (uint32_t c = 0; c < n_cons; ++c) {
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
    if constexpr (GDN_PREP_M_BETA != 0) {
        // The diag(beta) pages hold zeros off the diagonal: zeroed once, push_chunk_coefs stores only the diagonal.
        CircularBuffer coef_buf(cb_tok_coef);
        noc.async_write_zeros(
            coef_buf, GDN_CHUNK_COEF_PAGES * GDN_TOK_COEF_DEPTH * STATE_TILE_BYTES, {.offset_bytes = 0});
        noc.write_zeros_l1_barrier();
    }
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
    // NORM_SPLIT: the norm units' o landings and output stores are served while a prep row waits for its grant or for
    // compute, and after the last prep row.
    const uint32_t all_units = (last_tr - first_tr + 1 - TAIL) * RATIO;
    const uint32_t norm_units = norm_offset < all_units ? (all_units - 1 - norm_offset) / norm_stride + 1 : 0;
    uint32_t landed_units = 0;
    uint32_t stored_units = 0;
    uint32_t head_landed[RATIO] = {};  // NORM_SPLIT: this core's landed units per head
    volatile tt_l1_ptr uint32_t* granted_count =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(GDN_LANDED_FIRST_SEM_ID + INBOX_SLOTS));
    auto serve_norms = [&]() {
        if (landed_units < norm_units) {
            // One cumulative counter per (head, piece): a state core sends this core's units of its head in order,
            // so the unit has landed when every piece of its head has sent one more than the head's earlier units.
            const uint32_t j = (norm_offset + landed_units * norm_stride) % RATIO;
            bool landed = true;
            for (uint32_t p = 0; p < PIECES; ++p) {
                volatile tt_l1_ptr uint32_t* o_count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(GDN_LANDED_FIRST_SEM_ID + j * PIECES + p));
                landed = landed && *o_count > head_landed[j];
            }
            if (landed) {
                cb_push_back(cb_o_in, Vt);
                ++head_landed[j];
                ++landed_units;
            }
        }
        if (stored_units < landed_units && cb_pages_available_at_front(cb_out_rows, Vt)) {
            const uint32_t u = norm_offset + stored_units * norm_stride;
            const uint32_t tr = first_tr + u / RATIO;
            store_rows(
                out_acc,
                Vt,
                tr * out_width_tiles + (head0 + u % RATIO) * Vt,
                tr * TILE_ROWS,
                row_lo(tr),
                row_hi(tr));
            ++stored_units;
        }
    };
    auto wait_send = [&](uint32_t pages) {
        if constexpr (NORM_SPLIT) {
            // At least once per part: a row's sends take long where many rows land at once, and compute may be
            // waiting for an o (norm_mid).
            do {
                serve_norms();
            } while (!cb_pages_available_at_front(cb_prep_send, pages));
        }
        cb_wait_front(cb_prep_send, pages);
    };
    const uint32_t first_row = first_tr + tr_offset;
    uint32_t sent = 0;
    for (uint32_t tr = first_row; tr <= last_tr; tr += tr_stride, ++sent) {
        GDN_ZONE("gdn_wr_prep_row");
        const uint32_t base_row = tr * TILE_ROWS;
        const uint32_t lo = row_lo(tr);
        const uint32_t hi = row_hi(tr);
        // NORM_SPLIT: compute may be in a norm unit ahead of this row's heads (norm_mid), which needs its o landed.
        if constexpr (NORM_SPLIT) {
            while (!cb_pages_available_at_front(cb_gates, 2)) {
                serve_norms();
            }
        }
        cb_wait_front(cb_gates, 2);
        volatile tt_l1_ptr uint32_t* gates = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_gates));
        for (uint32_t j = 0; j < RATIO; ++j) {
            if constexpr (NORM_SPLIT) {
                while (!cb_pages_reservable_at_back(cb_tok_coef, GDN_CHUNK_COEF_PAGES)) {
                    serve_norms();
                }
            }
            push_chunk_coefs(gates, (head0 + j) % TILE_ROWS, base_row, lo, hi);
        }
        cb_pop_front(cb_gates, 2);
        if constexpr (!NORM_SPLIT) {
            if (sent >= 1) {
                land_o(sent);
            }
            if (sent >= 2) {
                store(tr - 2 * tr_stride);
            }
        }

        // Every state core of the group grants each of its rows once, in order; this core's next row is granted
        // only after this one has landed (stride >= inbox slots), so the cumulative count identifies the row.
        if constexpr (NORM_SPLIT) {
            while (*granted_count < (sent + 1) * n_cons) {
                serve_norms();
            }
        }
        granted.wait_min((sent + 1) * n_cons);
        GDN_MARK("m_wr_granted");
        // The row's o slot is free (compute has normalized the row that used it), so the row may be sent.
        // NORM_SPLIT: a norm core's cb_o_in holds every row it normalizes (host), so no row waits for a slot.
        if constexpr (!NORM_SPLIT) {
            cb_reserve_back(cb_o_in, O_PAGES);
        }
        const uint32_t slot = (tr - first_tr) % INBOX_SLOTS;
        // One head block at a time, each part sent as compute pushes it (pack order: exp(c) | QD, M | kq | knT | v).
        for (uint32_t j = 0; j < RATIO; ++j) {
            wait_send(VEC_PAGES + GDN_IN_MAT_PAGES);
            const uint32_t vec_src = get_read_ptr(cb_prep_send);
            const uint32_t mat_src = vec_src + VEC_PAGES * T_BYTES;
            const uint32_t kq_src = mat_src + GDN_IN_MAT_PAGES * T_BYTES;
            const uint32_t knT_src = kq_src + 2 * Kt * T_BYTES;
            const uint32_t v_src = knT_src + Kt * T_BYTES;
            mcast_rects(head_rects[j], vec_src, vec_base + slot * VEC_PAGES * T_BYTES, VEC_PAGES * T_BYTES);
            mcast_rects(
                head_rects[j], mat_src, mat_base + slot * GDN_IN_MAT_PAGES * T_BYTES, GDN_IN_MAT_PAGES * T_BYTES);
            wait_send(VEC_PAGES + GDN_IN_MAT_PAGES + 2 * Kt);
            mcast_rects(head_rects[j], kq_src, kq_base + slot * 2 * Kt * T_BYTES, 2 * Kt * T_BYTES);
            wait_send(HEAD_PAGES - Vt);
            mcast_rects(head_rects[j], knT_src, knT_base + slot * Kt * T_BYTES, Kt * T_BYTES);
            wait_send(HEAD_PAGES);
            for (uint32_t p = 0; p < PIECES; ++p) {
                const uint32_t c = j * PIECES + p;
                noc_async_write(
                    v_src + p * Vb * T_BYTES,
                    get_noc_addr(cons_x[c], cons_y[c], v_base + slot * Vb * T_BYTES),
                    Vb * T_BYTES);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_prep_send, HEAD_PAGES);
            GDN_MARK("m_wr_head");
            if constexpr (GDN_PREP_HEAD_UP != 0) {
                // The head's pieces step the row as soon as their block is in, not after the row's last head.
                for (uint32_t c0 = j * PIECES; c0 < n_cons; c0 += GROUP) {
                    for (uint32_t c = c0; c < c0 + PIECES; ++c) {
                        Semaphore<> landed(GDN_LANDED_FIRST_SEM_ID + slot);
                        landed.up(noc, cons_x[c], cons_y[c], 1);
                    }
                }
            }
        }
        for (uint32_t c = 0; GDN_PREP_HEAD_UP == 0 && c < n_cons; ++c) {
            Semaphore<> landed(GDN_LANDED_FIRST_SEM_ID + slot);
            landed.up(noc, cons_x[c], cons_y[c], 1);
        }
        GDN_MARK("m_wr_row_up");
    }
    if constexpr (NORM_SPLIT) {
        while (stored_units < norm_units) {
            serve_norms();
        }
        return;
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
        GDN_MARK("m_wr_sent");
    }
    if (!q_injector && receive) {
        q_channel.optional_receiver(noc)->receive();
        GDN_MARK("m_wr_q");
    }
    if (!k_injector && receive) {
        k_channel.optional_receiver(noc)->receive();
        GDN_MARK("m_wr_k");
    }
    cb_push_back(cb_kq_rows, 2 * Kt);
    cb_push_back(cb_kq_rows_f32, 2 * Kt);
}

}  // namespace

// Issues the piece's Kt x Vb state tiles of cb_state_out to recurrent_state_out (pair state page `first_page` on),
// Final-state tiles of this core that are issued and not acknowledged; current only while the command buffer is free.
FORCE_INLINE uint32_t state_tiles_in_flight() {
    return NOC_STATUS_READ_REG(noc_index, NIU_MST_REQS_OUTSTANDING_ID(STATE_TRID)) +
           NOC_STATUS_READ_REG(noc_index, NIU_MST_REQS_OUTSTANDING_ID(STATE_TRID + 1));
}

// except the reader's (reader_share); the caller barriers and calls pop_state.
template <uint32_t Kt, uint32_t Vt, uint32_t Vb, typename Accessor>
void store_state(
    const Accessor& state_out, uint32_t first_page, uint32_t unit_idx, uint32_t reader_units, uint32_t reader_share) {
    // Compute pushes the final state in update batches: each tile goes out as soon as it is packed.
    const uint32_t state_base = get_read_ptr(cb_state_out);
    if constexpr (STATE_WINDOW != 0) {
        noc_async_write_set_trid(STATE_TRID);
    }
    for (uint32_t j = 0; j < Kt * Vb; ++j) {
        if (gdn::state_tile_on_reader(unit_idx, j, reader_units, reader_share)) {
            continue;
        }
        cb_wait_front(cb_state_out, j + 1);
        if constexpr (STATE_WINDOW != 0) {
            while (!noc_cmd_buf_ready(noc_index, write_cmd_buf) || state_tiles_in_flight() >= STATE_WINDOW) {
            }
        }
        const uint32_t page = gdn::state_out_page<Kt, Vt, Vb>(first_page, j, STATE_STAGGER);
        noc_async_write(state_base + j * STATE_TILE_BYTES, state_out.get_noc_addr(page), STATE_TILE_BYTES);
    }
    if constexpr (STATE_WINDOW != 0) {
        noc_async_write_set_trid(0);
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
    // QK_RAW: the inverse norms of q and k.
    constexpr auto qr_channel = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("qkqr_ct_offset"), get_named_compile_time_arg_val("qkqr_rt_offset")>();
    constexpr auto kr_channel = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("qkkr_ct_offset"), get_named_compile_time_arg_val("qkkr_rt_offset")>();
    constexpr auto state_out_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto state_in_args =
        TensorAccessorArgs<state_out_args.next_compile_time_args_offset()>();
#if GDN_SMALL_MCAST  // the small inputs' accessors and channel exist only in SMALL_MCAST programs
    // The small inputs a q core reads for its key head's state cores (common args 3..7).
    constexpr auto a_log_args = TensorAccessorArgs<state_in_args.next_compile_time_args_offset()>();
    constexpr auto dt_bias_args = TensorAccessorArgs<a_log_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<dt_bias_args.next_compile_time_args_offset()>();
    constexpr auto a_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto norm_w_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto small_channel = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("qks_ct_offset"), get_named_compile_time_arg_val("qks_rt_offset")>();
#endif

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
    // R-PREP state core: tile row i's o goes to prep core i % producers; NORM_SPLIT: the o of norm unit
    // u = i * RATIO + head goes to slot u / norm_count of prep core norm_first + u % norm_count.
    const uint32_t norm_first = STATE ? get_arg_val<uint32_t>(arg++) : 0;
    const uint32_t norm_count = STATE ? get_arg_val<uint32_t>(arg++) : 1;
    // SEQ_SPLIT: the other half's cores of the head (a P core's destinations), then the next segment's core.
    uint32_t seg_peer_x[PIECES];
    uint32_t seg_peer_y[PIECES];
    uint32_t seg_next_x = 0, seg_next_y = 0;
    if constexpr (STATE && SPLIT) {
        for (uint32_t p = 0; p < PIECES; ++p) {
            seg_peer_x[p] = get_arg_val<uint32_t>(arg++);
            seg_peer_y[p] = get_arg_val<uint32_t>(arg++);
        }
        seg_next_x = get_arg_val<uint32_t>(arg++);
        seg_next_y = get_arg_val<uint32_t>(arg++);
    }
    // STATE_SPLIT: the reader's final-state tiles (gdn::state_tile_on_reader).
    constexpr bool split_args_role = !QK_CORE && !PREP && !PREP_QK && !STATE;  // R-QKCORE pair / local cores
    const bool split_args = STATE_SPLIT && split_args_role;
    const uint32_t reader_units = split_args ? get_arg_val<uint32_t>(arg++) : 0;
    const uint32_t reader_share = split_args ? get_arg_val<uint32_t>(arg++) : 0;
    const bool reader_writes = reader_units != 0 || reader_share != 0;

    // SMALL_MCAST (q core): the key head's gate column tile of A_log, dt_bias, b and a, then norm_weight, read first
    // (before the state cores' S0 reads queue at the banks) under their own transaction id, so that the q / k window
    // gather never waits for them, and multicast to the key head's state cores once landed.
    constexpr uint32_t SMALL_TRID = 1;
    constexpr uint32_t QK_WINDOW_TRID = 2;
    const bool small_sender = QK_CORE && GDN_SMALL_MCAST && qk_role == GDN_QK_Q_INJECTOR;
    const uint32_t small_base = GDN_SMALL_MCAST ? get_write_ptr(GDN_CB_SMALL_IN) : 0;
#if GDN_SMALL_MCAST
    if constexpr (QK_CORE) {
        if (small_sender) {
            const uint32_t gate_page = first_unit * RATIO / TILE_ROWS;  // decode: tile row 0
            // Packed form: b_args and a_args are mixed_qkv's; its gate tile lands in both slots, b is split below.
            const uint32_t ba_page = GDN_PACKED_GATE_TILE + gate_page;
            const uint64_t gate_srcs[GDN_SMALL_IN_GATE_PAGES] = {
                TensorAccessor(a_log_args, get_common_arg_val<uint32_t>(3), OUT_TILE_BYTES).get_noc_addr(gate_page),
                TensorAccessor(dt_bias_args, get_common_arg_val<uint32_t>(4), OUT_TILE_BYTES).get_noc_addr(gate_page),
                TensorAccessor(b_args, get_common_arg_val<uint32_t>(5), OUT_TILE_BYTES).get_noc_addr(ba_page),
                TensorAccessor(a_args, get_common_arg_val<uint32_t>(6), OUT_TILE_BYTES).get_noc_addr(ba_page)};
            noc_async_read_set_trid(SMALL_TRID);
            uint32_t dst = small_base;
            for (const uint64_t src : gate_srcs) {
                noc_async_read(src, dst, OUT_TILE_BYTES);
                dst += OUT_TILE_BYTES;
            }
            const auto norm_w = TensorAccessor(norm_w_args, get_common_arg_val<uint32_t>(7), OUT_TILE_BYTES);
            for (uint32_t vt = 0; vt < Vt; ++vt, dst += OUT_TILE_BYTES) {
                noc_async_read(norm_w.get_noc_addr(vt), dst, OUT_TILE_BYTES);
            }
            noc_async_read_set_trid(QK_WINDOW_TRID);
        }
    }
#endif

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

    // S0_WRITER: this RISC, idle until the first gates, reads every unit's S0 on its own NoC after the window zeroing
    // (or, AFTER_INPUTS, once the reader's first weights / gate rows have landed); the host sets it only where
    // cb_state_in holds every unit. Unit u's reads carry transaction id u + 1; its S0 goes to compute after the unit's
    // coefficients and q / k rows.
    const bool s0_by_writer = GDN_S0_WRITER && split_args_role && get_arg_val<uint32_t>(arg++) != 0;
    if (s0_by_writer) {
        const auto state_in = TensorAccessor(state_in_args, get_common_arg_val<uint32_t>(2), STATE_TILE_BYTES);
        if constexpr (GDN_S0_WRITER_AFTER_INPUTS) {
            cb_wait_front(GDN_CB_S0_GO, 1);
            cb_pop_front(GDN_CB_S0_GO, 1);
        }
        cb_reserve_back(cb_state_in, num_units * n_state);
        cb_reserve_back(cb_state_in_mm, num_units * n_state);
        const uint32_t dst = get_write_ptr(cb_state_in);
        for (uint32_t unit_idx = 0; unit_idx < num_units; ++unit_idx) {
            const uint32_t unit = first_unit + unit_idx * unit_stride;
            noc_async_read_set_trid(unit_idx + 1);
            gdn::read_state_slice<Kt, Vt, Vb, STATE_TILE_BYTES>(
                state_in,
                dst + unit_idx * n_state * STATE_TILE_BYTES,
                unit / PIECES,
                (unit % PIECES) * Vb,
                STATE_STAGGER,
                n_state);
        }
        noc_async_read_set_trid(0);
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
        const uint32_t hist_pct = qk_role == GDN_QK_Q_INJECTOR ? GDN_QK_Q_READER_PCT : 50;
        const uint32_t cs_tiles = (GDN_HISTORY * batch + TILE_ROWS - 1) / TILE_ROWS;
        const uint32_t cs_stage = get_write_ptr(cb_hist_stage) + Kt * OUT_TILE_BYTES;
        const uint32_t win = get_write_ptr(cb_window);
        constexpr uint32_t slot_bytes = GDN_CONV_TAPS * OUT_TILE_BYTES;
        constexpr uint32_t small_bytes = (GDN_SMALL_IN_GATE_PAGES + Vt) * OUT_TILE_BYTES;
        bool small_pending = small_sender;
        // Every column's slot zeroed while the staging reads are in flight (this RISC waits for them anyway).
        noc.async_write_zeros(window_buf, Kt * slot_bytes, {.offset_bytes = 0});
        noc.write_zeros_l1_barrier();
        cb_wait_front(cb_scratch, 1);  // staging landed
        for (uint32_t col = 0; col < Kt; ++col) {
            gdn::copy_history_rows<OUT_TILE_BYTES, GDN_HISTORY>(
                cs_stage + col * cs_tiles * OUT_TILE_BYTES,
                win + col * slot_bytes,
                gdn::history_split(batch, hist_pct),
                batch);
            if (small_sender) {
                noc_async_read_barrier_with_trid(QK_WINDOW_TRID);
            } else {
                noc_async_read_barrier();
            }
            cb_reserve_back(cb_window_zeroed, 1);
            cb_push_back(cb_window_zeroed, 1);
            GDN_MARK("m_wr_col");
            // The small inputs as soon as a column hand-off finds them landed, at the latest after the last column.
            if (small_pending &&
                (col + 1 == Kt || ncrisc_noc_read_with_transaction_id_flushed(noc_index, SMALL_TRID))) {
                noc_async_read_barrier_with_trid(SMALL_TRID);
                noc_async_read_set_trid(0);
                if constexpr (GDN_PACKED_TILES != 0) {
                    const uint32_t b_tile = small_base + 2 * OUT_TILE_BYTES;
                    gdn::packed_gate_b<OUT_TILE_BYTES>(b_tile, b_tile, Hv, 0, batch);
                }
#if GDN_SMALL_MCAST
                small_channel.optional_sender(noc)->send(small_base, small_base, small_bytes);
#endif
                small_pending = false;
                GDN_MARK("m_wr_small_sent");
            }
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
        GDN_MARK("m_wr_sent");
        if constexpr (QK_RAW) {
            // The inverse norm lands in tile 0 (k) or 1 (q) of the state cores' cb_rinv_in, also at its base.
            cb_wait_front(cb_rinv, 1);
            const uint32_t rinv_dst = get_write_ptr(cb_rinv_in);
            if (qk_role == GDN_QK_Q_INJECTOR) {
                qr_channel.optional_sender(noc)->send(
                    get_read_ptr(cb_rinv), rinv_dst + STATE_TILE_BYTES, STATE_TILE_BYTES);
            } else {
                kr_channel.optional_sender(noc)->send(get_read_ptr(cb_rinv), rinv_dst, STATE_TILE_BYTES);
            }
            cb_pop_front(cb_rinv, 1);
            GDN_MARK("m_wr_sent_rinv");
        }
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
    // WRITER_ASYNC: the loop runs unit 0 only; async_units serves the rest.
    const bool async = WRITER_ASYNC && lag && !reader_writes && !s0_by_writer && PIECES == 1;
    for (uint32_t unit_idx = 0; unit_idx < (async ? 1 : num_units); ++unit_idx) {
        const uint32_t unit = first_unit + unit_idx * unit_stride;
        GDN_ZONE("gdn_wr_unit");
        const uint32_t pair = unit / PIECES;
        const uint32_t piece = unit % PIECES;
        const uint32_t v_first = piece * Vb;
        const bool p_half = SPLIT && pair / Hv >= SEGS;
        const uint32_t seq = p_half ? pair / Hv - (SEGS - 1) : pair / Hv;
        const uint32_t head = pair % Hv;
        const uint32_t gate_col = head % TILE_ROWS;
        const uint32_t seq_lo = seq * T;
        const uint32_t seq_hi = seq_lo + T;
        const uint32_t first_tr = seq_lo / TILE_ROWS;
        const uint32_t last_tr = (seq_hi - 1) / TILE_ROWS;
        // SEQ_SPLIT: the final state of a state core (a segment 1..'s after its combine, before its corrected rows).
        auto seg_store = [&]() {
            if (seq + 1 == SEGS) {
                store_state<Kt, Vt, Vb>(state_out, head * Kt * Vt + v_first, unit_idx, reader_units, reader_share);
            } else {
                const auto& in_if = get_local_cb_interface(cb_seg_in);
                const uint32_t dst = in_if.fifo_limit - in_if.fifo_size + Kt * Kt * STATE_TILE_BYTES;
                cb_wait_front(cb_state_out, n_state);
                noc_async_write(
                    get_read_ptr(cb_state_out), get_noc_addr(seg_next_x, seg_next_y, dst), n_state * STATE_TILE_BYTES);
                noc_async_write_barrier();
                Noc noc;
                Semaphore<>(SEG_SEM_ID).up(noc, seg_next_x, seg_next_y, 1);
            }
            noc_async_write_barrier();
            pop_state(n_state, reader_writes);
        };
        if constexpr (STATE && SPLIT) {
            if (p_half) {
                // The piece of P (state columns [v_first, v_first + Vb) of the Kt x Kt tiles), then Oh_P row by row.
                const auto& in_if = get_local_cb_interface(cb_seg_in);
                const auto& row_if = get_local_cb_interface(cb_seg_row);
                const uint32_t in_base = in_if.fifo_limit - in_if.fifo_size;
                const uint32_t row_base = row_if.fifo_limit - row_if.fifo_size;
                Noc noc;
                cb_wait_front(cb_state_out, n_state);
                const uint32_t p_src = get_read_ptr(cb_state_out);
                for (uint32_t q = 0; q < PIECES; ++q) {
                    for (uint32_t kt = 0; kt < Kt; ++kt) {
                        noc_async_write(
                            p_src + kt * Vb * STATE_TILE_BYTES,
                            get_noc_addr(
                                seg_peer_x[q], seg_peer_y[q], in_base + (kt * Kt + v_first) * STATE_TILE_BYTES),
                            Vb * STATE_TILE_BYTES);
                    }
                }
                noc_async_write_barrier();
                for (uint32_t q = 0; q < PIECES; ++q) {
                    Semaphore<>(SEG_SEM_ID).up(noc, seg_peer_x[q], seg_peer_y[q], 1);
                }
                cb_pop_front(cb_state_out, n_state);
                for (uint32_t i = 0; i <= last_tr - first_tr; ++i) {
                    for (uint32_t q = 0; q < PIECES; ++q) {
                        Semaphore<>(SEG_SEM_ID + q).wait_min(i + 1);
                    }
                    cb_wait_front(cb_reads, Vb);
                    const uint32_t dst = row_base + ((i % GDN_SEG_ROW_DEPTH) * Kt + v_first) * STATE_TILE_BYTES;
                    for (uint32_t q = 0; q < PIECES; ++q) {
                        noc_async_write(
                            get_read_ptr(cb_reads),
                            get_noc_addr(seg_peer_x[q], seg_peer_y[q], dst),
                            Vb * STATE_TILE_BYTES);
                    }
                    noc_async_write_barrier();
                    for (uint32_t q = 0; q < PIECES; ++q) {
                        Semaphore<>(SEG_SEM_ID + 1 + piece).up(noc, seg_peer_x[q], seg_peer_y[q], 1);
                    }
                    cb_pop_front(cb_reads, Vb);
                }
                continue;
            }
            if (seq > 0) {
                seg_store();
            }
        }

        for (uint32_t tr = first_tr + tr_offset; tr <= last_tr; tr += tr_stride) {
            const uint32_t base_row = tr * TILE_ROWS;
            const uint32_t lo = seq_lo > base_row ? seq_lo : base_row;
            const uint32_t hi = seq_hi < base_row + TILE_ROWS ? seq_hi : base_row + TILE_ROWS;

            // An injector sends its qn / kn first: its compute makes the gates after them. QK_RAW: y lands ahead of the
            // gates too, so compute's transposes and state read are not held behind them; the inverse norms after.
            const bool lands = !CHUNKED && qk_role != GDN_QK_OWN && !(hold_qk && unit_idx > 0);
            const bool injector = qk_role == GDN_QK_Q_INJECTOR || qk_role == GDN_QK_K_INJECTOR;
            const bool land_early = injector || QK_RAW;
            if (lands && land_early) {
                land_qk_rows(q_channel, k_channel, qk_role, Kt, unit_idx == 0);
                GDN_MARK("m_wr_landed");
            }

            // ---- per-token coefficient expansion (R-CHUNK: per-tile-row g / beta columns) ----
            if constexpr (!STATE) {
                cb_wait_front(cb_gates, 2);
                volatile tt_l1_ptr uint32_t* gates =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_gates));
                if constexpr (CHUNKED) {
                    push_chunk_coefs(gates, gate_col, base_row, lo, hi);
                }
                for (uint32_t row = lo; row < hi && !CHUNKED; ++row) {
                    push_token_coefs(gates, row - base_row, gate_col);
                }
                cb_pop_front(cb_gates, 2);
                GDN_MARK("m_wr_coef");
            }

            if (lands && !land_early) {
                land_qk_rows(q_channel, k_channel, qk_role, Kt, unit_idx == 0);
                GDN_MARK("m_wr_landed");
            }
            if (s0_by_writer) {
                // After the unit's coefficients and q / k rows: compute uses those before S0.
                noc_async_read_barrier_with_trid(unit_idx + 1);
                cb_push_back(cb_state_in, n_state);
                cb_push_back(cb_state_in_mm, n_state);
                GDN_MARK("m_wr_s0");
            }
            if (QK_RAW && lands) {
                // The inverse norms of the key head, once per core (held for its later units like y).
                Noc noc;
                cb_reserve_back(cb_rinv_in, 2);
                qr_channel.optional_receiver(noc)->receive();
                kr_channel.optional_receiver(noc)->receive();
                cb_push_back(cb_rinv_in, 2);
                GDN_MARK("m_wr_rinv");
            }

            // ---- partial sum o^2 all-gather (R-VSPLIT) ----
            // Each piece writes its partial into slot `piece` of the gather buffer on every piece core
            // (same L1 address everywhere: identical CB layout, identical tile-row count), then bumps
            // the peers' counters. cb_osum_gather holds two tile rows: a peer can be at most one tile
            // row ahead, because it cannot finish row i+1 without this core's partial of row i+1, which
            // compute produces only after it has consumed row i's slots.
            const bool tail_row = STATE && tr + TAIL > last_tr;
            if (tail_row && tr + TAIL == last_tr + 1) {
                // The final state first: the tail rows' norm follows the chain.
                store_state<Kt, Vt, Vb>(state_out, pair * Kt * Vt + v_first, unit_idx, reader_units, reader_share);
                noc_async_write_barrier();
                pop_state(n_state, reader_writes);
            }
            if (STATE && !tail_row) {
                // ---- R-PREP: o of the piece into its place in slot k % GDN_O_IN_DEPTH of the row's prep core ----
                const uint32_t i = tr - first_tr;
                const uint32_t head_in_group = head % RATIO;
                const uint32_t u = i * RATIO + head_in_group;
                const uint32_t q = NORM_SPLIT ? norm_first + u % norm_count : i % producers;
                const uint32_t o_tile = NORM_SPLIT ? (u / norm_count) * Vt
                                                   : ((i / producers) % GDN_O_IN_DEPTH) * RATIO * Vt + head_in_group * Vt;
                const uint32_t dst = cb_region_base(cb_in_kq, GDN_OFFSET_IN_KQ) + GDN_OFFSET_O_IN +
                                     (o_tile + v_first) * STATE_TILE_BYTES;
                cb_wait_front(cb_o_acc, Vb);
                noc_async_write(
                    get_read_ptr(cb_o_acc), get_noc_addr(producer_x[q], producer_y[q], dst), Vb * STATE_TILE_BYTES);
                noc_async_write_barrier();
                cb_pop_front(cb_o_acc, Vb);
                Noc noc;
                Semaphore<> o_landed(GDN_LANDED_FIRST_SEM_ID + (NORM_SPLIT ? head_in_group * PIECES + v_first / Vb : INBOX_SLOTS + 1));
                o_landed.up(noc, producer_x[q], producer_y[q], 1);
                continue;
            }
            if constexpr (PIECES > 1) {
                Noc noc;
                Semaphore<> osum_landed(OSUM_SEM_ID);
                cb_wait_front(cb_osum_part, 1);
                GDN_MARK("m_wr_osum_part");
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
                GDN_MARK("m_wr_osum");
            }
            if constexpr (STATE) {
                store_rows(out_acc, Vb, tr * out_width_tiles + head * Vt + v_first, base_row, lo, hi);
                continue;
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
            if (SPLIT) {
                if (seq == 0) {
                    seg_store();
                }
                continue;
            }
            if (TAIL != 0) {
                continue;
            }
            store_state<Kt, Vt, Vb>(state_out, pair * Kt * Vt + v_first, unit_idx, reader_units, reader_share);
            noc_async_write_barrier();
            pop_state(n_state, reader_writes);
        }
    }
    if (async) {
        // Units 1 on, T = 1: three jobs polled in turn, none of which waits. (1) The next unit's coefficients as soon
        // as compute has its gates. (2) Final states tile by tile as compute packs them and the write command buffer
        // is free; up to two states in flight (transaction id = state parity), each freed when its own writes have
        // completed. (3) A unit's output rows once normalized, one unit in flight. A write-starved core thus keeps
        // computing until cb_state_out is full.
        const auto& state_cb = get_local_cb_interface(cb_state_out);
        const uint32_t state_slots = state_cb.fifo_size / (n_state * STATE_TILE_BYTES) < 2 ? 1 : 2;
        uint32_t coef_unit = 1;
        uint32_t state_unit = 0, state_tile = 0, states_freed = 0;
        uint32_t rows_unit = 0, rows_freed = 0;
        auto pair_of = [&](uint32_t unit_idx) { return (first_unit + unit_idx * unit_stride) / PIECES; };
        while (states_freed < num_units || rows_freed < num_units) {
            if (coef_unit < num_units && cb_pages_available_at_front(cb_gates, 2) &&
                cb_pages_reservable_at_back(cb_tok_coef, GDN_TOK_COEF_PAGES)) {
                const uint32_t pair = pair_of(coef_unit);
                push_token_coefs(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_gates)),
                    (pair / Hv) % TILE_ROWS,
                    (pair % Hv) % TILE_ROWS);
                cb_pop_front(cb_gates, 2);
                ++coef_unit;
                GDN_MARK("m_wr_coef");
            }
            const bool cmd_free = noc_cmd_buf_ready(noc_index, write_cmd_buf);
            const uint32_t ahead = (state_unit - states_freed) * n_state + state_tile;  // tiles before the next one
            // A non-last state starts after the next unit's coefficients, as in the blocking order (earlier
            // write-backs delay the other cores' S0 reads: decode B=16 +4.4 us).
            const bool state_due = state_unit + 1 == num_units || coef_unit > state_unit + 1 || !ASYNC_STATE_HOLD;
            const bool window_open = STATE_WINDOW == 0 || (cmd_free && state_tiles_in_flight() < STATE_WINDOW);
            if (cmd_free && state_unit < num_units && state_due && state_unit - states_freed < state_slots &&
                window_open && cb_pages_available_at_front(cb_state_out, ahead + 1)) {
                uint32_t src = get_read_ptr(cb_state_out) + ahead * STATE_TILE_BYTES;
                src -= src >= state_cb.fifo_limit ? state_cb.fifo_size : 0;
                const uint32_t page =
                    gdn::state_out_page<Kt, Vt, Vb>(pair_of(state_unit) * Kt * Vt, state_tile, STATE_STAGGER);
                noc_async_write_set_trid(STATE_TRID + (state_unit & 1));
                noc_async_write(src, state_out.get_noc_addr(page), STATE_TILE_BYTES);
                if (++state_tile == n_state) {
                    state_tile = 0;
                    ++state_unit;
                    GDN_MARK("m_wr_state_issued");
                }
            } else if (cmd_free && rows_unit < num_units && rows_unit == rows_freed &&
                       cb_pages_available_at_front(cb_out_rows, Vb)) {
                const uint32_t pair = pair_of(rows_unit);
                const uint32_t seq = pair / Hv;
                const uint32_t base_row = seq / TILE_ROWS * TILE_ROWS;
                noc_async_write_set_trid(ROWS_TRID);
                store_rows<false>(
                    out_acc, Vb, seq / TILE_ROWS * out_width_tiles + (pair % Hv) * Vt, base_row, seq, seq + 1);
                ++rows_unit;
            } else if (cmd_free) {
                // The id counters are current only once the last command has left its buffer.
                if (states_freed < state_unit &&
                    ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, STATE_TRID + (states_freed & 1))) {
                    cb_pop_front(cb_state_out, n_state);
                    ++states_freed;
                    GDN_MARK("m_wr_state_done");
                }
                if (rows_freed < rows_unit &&
                    ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, ROWS_TRID)) {
                    cb_pop_front(cb_out_rows, Vb);
                    ++rows_freed;
                }
            }
        }
        noc_async_write_set_trid(0);
        noc_async_write_barrier();
        pending_state = 0xffffffffu;
    } else if (lag) {
        // The previous rows' writes must have left cb_out_rows before the last unit's rows are read from its front.
        noc_async_writes_flushed();
        cb_pop_front(cb_out_rows, Vb);
        retiring = false;
        store_rows<false>(out_acc, Vb, lag_page, lag_base, lag_lo, lag_hi);
        retiring = true;
    }
    retire();
    GDN_MARK("m_wr_done");
}
