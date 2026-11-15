// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gated_delta_net reader (NCRISC).
//
// Per unit (pair (b, h), piece) of this core's contiguous range; a piece owns Vb = Vt / PIECES of the
// pair's v / state columns (R-VSPLIT; R-STEP is PIECES = 1) and every q/k column of its key head:
//   load_pair_block: conv_weight tiles (raw for the FPU conv, else row j -> row 0 of tap tile j), A_log / dt_bias
//                    gate-column tiles, norm_weight tiles.
//   per tile row of the pair: b/a gate tiles, the conv window (4 tap tiles per column tile, rows
//                    outside the pair's token rows zero), S0 after the first tile row's prep
//                    inputs (stall shadow), z tiles.
//   copy_conv_state_block: conv_state_out rows 3b..3b+2 of the pair's columns (bit-exact DM copy).
// Token rows are moved as face-row segments (64 B fp32, 32 B bf16; bf16 reads go through RowReader
// staging) unless the pair owns the whole tile row.

#include <stdint.h>

#include "api/alignment.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {

// CB indices and GDN_* spec constants are host defines (gated_delta_net_program_descriptor.py).
constexpr uint32_t cb_state_in = GDN_CB_STATE_IN;
constexpr uint32_t cb_state_in_mm = GDN_CB_STATE_IN_MM;
constexpr uint32_t cb_conv_w = GDN_CB_CONV_W;
constexpr uint32_t cb_window = GDN_CB_WINDOW;
constexpr uint32_t cb_gate_in = GDN_CB_GATE_IN;
constexpr uint32_t cb_gate_w = GDN_CB_GATE_W;
constexpr uint32_t cb_z = GDN_CB_Z;
constexpr uint32_t cb_norm_w = GDN_CB_NORM_W;
constexpr uint32_t cb_reduce_scaler = GDN_CB_REDUCE_SCALER;
constexpr uint32_t cb_scratch = GDN_CB_SCRATCH;
constexpr uint32_t cb_row_stage = GDN_CB_ROW_STAGE;
constexpr uint32_t cb_window_zeroed = GDN_CB_WINDOW_ZEROED;
constexpr uint32_t cb_hist_stage = GDN_CB_HIST_STAGE;
constexpr uint32_t cb_tri_mask = GDN_CB_TRI_MASK;

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t TILE_BYTES = get_compile_time_arg_val(6);        // token-tensor tile (dtype-derived on host)
constexpr uint32_t STATE_TILE_BYTES = get_compile_time_arg_val(7);  // fp32 state tile
constexpr uint32_t READ_ALIGN = get_compile_time_arg_val(8);        // DRAM read alignment
constexpr uint32_t STAGE_SLOTS = get_compile_time_arg_val(9);       // 0: face rows are read directly
constexpr bool CONV_W_RAW = get_compile_time_arg_val(10) != 0;      // raw conv_weight tiles (FPU conv)
constexpr uint32_t WINDOW_GROUP_COLS = get_compile_time_arg_val(11);  // column tiles per window hand-off
constexpr uint32_t WINDOW_SLOTS = get_compile_time_arg_val(12);       // window groups held by cb_window
constexpr uint32_t PIECES = get_compile_time_arg_val(13);             // column pieces per pair
constexpr bool CHUNKED = get_compile_time_arg_val(14) != 0;           // R-CHUNK: chunk step per tile row
constexpr uint32_t FACE_BYTES = TILE_BYTES / 4;                     // one 16x16 face
constexpr uint32_t SEG_BYTES = FACE_BYTES / 16;                     // one face row
constexpr uint32_t CS_COLS_PER_FLUSH = TILE_ROWS / GDN_HISTORY;     // conv_state_out columns per scratch tile
static_assert((STAGE_SLOTS == 0) == (SEG_BYTES % READ_ALIGN == 0), "row staging iff a face row is sub-alignment");

// Byte offset of the left half of tile row p (faces 0/2); the right half is FACE_BYTES further.
FORCE_INLINE uint32_t row_offset(uint32_t p) { return (p >> 4) * 2 * FACE_BYTES + (p & 15) * SEG_BYTES; }

// Moves single tile rows (two face-row segments) from a tensor page into an L1 tile.
// When a segment is narrower than the DRAM read alignment, each segment is read as the aligned chunk
// holding it into a staging slot and then copied L1-to-L1 (L1 alignment only); the copies are issued
// in batches of STAGE_SLOTS. finish() must precede any use of the destination tiles.
class RowReader {
public:
    explicit RowReader(uint32_t stage_base) : stage_(align_power_of_2(stage_base, READ_ALIGN)) {}

    template <typename Accessor>
    __attribute__((noinline)) void read(
        const Accessor& acc, uint32_t page, uint32_t src_row, uint32_t dst_tile, uint32_t dst_row) {
        const uint32_t src_off = row_offset(src_row);
        const uint32_t dst = dst_tile + row_offset(dst_row);
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t off = src_off + half * FACE_BYTES;
            if constexpr (STAGE_SLOTS == 0) {
                noc_async_read(acc.get_noc_addr(page, off), dst + half * FACE_BYTES, SEG_BYTES);
            } else {
                const uint32_t chunk_off = off & ~(READ_ALIGN - 1);
                const uint32_t slot = stage_ + pending_ * READ_ALIGN;
                noc_async_read(acc.get_noc_addr(page, chunk_off), slot, READ_ALIGN);
                staged_src_[pending_] = slot + (off - chunk_off);
                staged_dst_[pending_] = dst + half * FACE_BYTES;
                if (++pending_ == STAGE_SLOTS) {
                    finish();
                }
            }
        }
    }

    // Completes every outstanding read of this RISC, including staged rows.
    __attribute__((noinline)) void finish() {
        noc_async_read_barrier();
        if constexpr (STAGE_SLOTS != 0) {
            if (pending_ != 0) {
                for (uint32_t i = 0; i < pending_; ++i) {
                    noc_async_read(get_noc_addr(staged_src_[i]), staged_dst_[i], SEG_BYTES);
                }
                noc_async_read_barrier();
                pending_ = 0;
            }
        }
    }

private:
    static constexpr uint32_t SLOT_ARRAY = STAGE_SLOTS == 0 ? 1 : STAGE_SLOTS;
    uint32_t stage_;
    uint32_t pending_ = 0;
    uint32_t staged_src_[SLOT_ARRAY];
    uint32_t staged_dst_[SLOT_ARRAY];
};

// Rows [row_lo, row_hi) of a tensor page into the same rows of an L1 tile, for buffers whose other rows
// are never consumed (weights read by row index, gate tiles read per token row): each face-row span is
// widened to DRAM read alignment and read in place, so no staging copy is needed. Whole tile when every
// row is wanted. Completed by the next barrier / RowReader::finish().
template <typename Accessor>
__attribute__((noinline)) void read_row_span(
    const Accessor& acc, uint32_t page, uint32_t row_lo, uint32_t row_hi, uint32_t dst) {
    if (row_hi - row_lo == TILE_ROWS) {
        noc_async_read(acc.get_noc_addr(page), dst, TILE_BYTES);
        return;
    }
    constexpr uint32_t FACE_ROWS = TILE_ROWS / 2;
    for (uint32_t face_row0 = 0; face_row0 < TILE_ROWS; face_row0 += FACE_ROWS) {
        const uint32_t lo = row_lo > face_row0 ? row_lo : face_row0;
        const uint32_t hi = row_hi < face_row0 + FACE_ROWS ? row_hi : face_row0 + FACE_ROWS;
        if (lo >= hi) {
            continue;
        }
        const uint32_t begin = row_offset(lo) & ~(READ_ALIGN - 1);
        const uint32_t end = align_power_of_2(row_offset(hi - 1) + SEG_BYTES, READ_ALIGN);
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t off = begin + half * FACE_BYTES;
            noc_async_read(acc.get_noc_addr(page, off), dst + off, end - begin);
        }
    }
}

// Rows [src_row, src_row + n) of an L1 tile into rows [dst_row, dst_row + n) of another: one NoC copy per
// face half per run of rows that stays inside one face in both tiles (rows of a face are contiguous).
__attribute__((noinline)) void copy_local_rows(
    uint32_t src_tile, uint32_t src_row, uint32_t dst_tile, uint32_t dst_row, uint32_t n) {
    constexpr uint32_t FACE_ROWS = TILE_ROWS / 2;
    if (n == 1) {  // decode: one row per tap
        const uint32_t src = src_tile + row_offset(src_row);
        const uint32_t dst = dst_tile + row_offset(dst_row);
        noc_async_read(get_noc_addr(src), dst, SEG_BYTES);
        noc_async_read(get_noc_addr(src + FACE_BYTES), dst + FACE_BYTES, SEG_BYTES);
        return;
    }
    while (n != 0) {
        const uint32_t src_room = FACE_ROWS - (src_row & (FACE_ROWS - 1));
        const uint32_t dst_room = FACE_ROWS - (dst_row & (FACE_ROWS - 1));
        uint32_t run = src_room < dst_room ? src_room : dst_room;
        run = run < n ? run : n;
        const uint32_t src = src_tile + row_offset(src_row);
        const uint32_t dst = dst_tile + row_offset(dst_row);
        noc_async_read(get_noc_addr(src), dst, run * SEG_BYTES);
        noc_async_read(get_noc_addr(src + FACE_BYTES), dst + FACE_BYTES, run * SEG_BYTES);
        src_row += run;
        dst_row += run;
        n -= run;
    }
}

// Tensix-to-DRAM writes need only L1 alignment, so a face-row segment is written directly.
template <typename Accessor>
__attribute__((noinline)) void write_row(
    const Accessor& acc, uint32_t src_tile, uint32_t src_row, uint32_t page, uint32_t dst_row) {
    const uint32_t src = src_tile + row_offset(src_row);
    const uint32_t dst_off = row_offset(dst_row);
    noc_async_write(src, acc.get_noc_addr(page, dst_off), SEG_BYTES);
    noc_async_write(src + FACE_BYTES, acc.get_noc_addr(page, dst_off + FACE_BYTES), SEG_BYTES);
}

// Rows [lo, hi) of tile row tr (global row indices) of one tile column into an L1 tile.
template <typename Accessor>
__attribute__((noinline)) void read_tile_rows(
    RowReader& rows,
    const Accessor& acc,
    uint32_t page,
    uint32_t tile_row_base,
    uint32_t lo,
    uint32_t hi,
    uint32_t dst) {
    if (hi - lo == TILE_ROWS) {
        noc_async_read(acc.get_noc_addr(page), dst, TILE_BYTES);
    } else {
        for (uint32_t row = lo; row < hi; ++row) {
            rows.read(acc, page, row - tile_row_base, dst, row - tile_row_base);
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
    constexpr uint32_t Vb = Vt / PIECES;  // v / state column tiles of one piece
    constexpr uint32_t n_state = Kt * Vb;
    constexpr uint32_t z_width_tiles = Hv * Vt;
    constexpr uint32_t pair_cols = 2 * Kt + Vb;
    constexpr uint32_t conv_w_pages = CONV_W_RAW ? pair_cols : GDN_CONV_TAPS * pair_cols;

    constexpr auto mixed_args = TensorAccessorArgs<15>();
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
    const auto state = TensorAccessor(state_args, get_arg_val<uint32_t>(arg++), STATE_TILE_BYTES);
    const auto conv_w = TensorAccessor(conv_w_args, get_arg_val<uint32_t>(arg++), TILE_BYTES);
    const auto a_log = TensorAccessor(a_log_args, get_arg_val<uint32_t>(arg++), TILE_BYTES);
    const auto dt_bias = TensorAccessor(dt_bias_args, get_arg_val<uint32_t>(arg++), TILE_BYTES);
    const auto norm_w = TensorAccessor(norm_w_args, get_arg_val<uint32_t>(arg++), TILE_BYTES);
    const auto conv_state_out = TensorAccessor(conv_state_out_args, get_arg_val<uint32_t>(arg++), TILE_BYTES);
    const uint32_t first_unit = get_arg_val<uint32_t>(arg++);
    const uint32_t num_units = get_arg_val<uint32_t>(arg++);
    const uint32_t T = get_arg_val<uint32_t>(arg++);

    Noc noc;
    CircularBuffer window_buf(cb_window);
    // Tile-local rows [lo, hi) each cb_window slot may hold non-zero data in. Every write of a window
    // group lands in its token rows, so a slot needs re-zeroing only when the new group's rows do not
    // cover the old ones. The writer zeroes the whole buffer once at kernel start (empty range:
    // lo = TILE_ROWS, hi = 0).
    uint32_t dirty_lo[WINDOW_SLOTS];
    uint32_t dirty_hi[WINDOW_SLOTS];
    for (uint32_t slot = 0; slot < WINDOW_SLOTS; ++slot) {
        dirty_lo[slot] = TILE_ROWS;
        dirty_hi[slot] = 0;
    }
    uint32_t window_slot = 0;
    uint32_t zero_tokens = WINDOW_SLOTS;  // the writer zeroes slot k and then sends token k
    const uint32_t hist_stage = get_write_ptr(cb_hist_stage);
    RowReader rows(STAGE_SLOTS == 0 ? 0 : get_write_ptr(cb_row_stage));

    dataflow_kernel_lib::
        prepare_reduce_scaler<cb_reduce_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(1.0f);
    if constexpr (CHUNKED) {
        // Lower-triangular 0/1 mask (column <= row) of the chunk decay matrix, once per kernel: zeroed over
        // the NoC, then only the 528 ones are stored.
        CircularBuffer mask_buf(cb_tri_mask);
        cb_reserve_back(cb_tri_mask, 1);
        noc.async_write_zeros(mask_buf, STATE_TILE_BYTES, {.offset_bytes = 0});
        noc.write_zeros_l1_barrier();
        volatile tt_l1_ptr uint32_t* mask = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_tri_mask));
        constexpr uint32_t ONE_BITS = 0x3f800000u;
        for (uint32_t r = 0; r < TILE_ROWS; ++r) {
            for (uint32_t c = 0; c <= r; ++c) {
                mask[((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15)] = ONE_BITS;
            }
        }
        cb_push_back(cb_tri_mask, 1);
    }

    for (uint32_t unit = first_unit; unit < first_unit + num_units; ++unit) {
        const uint32_t pair = unit / PIECES;
        const uint32_t v_first = (unit % PIECES) * Vb;  // first v / state column tile of the piece
        const uint32_t seq = pair / Hv;
        const uint32_t head = pair % Hv;
        const uint32_t key_head = head / ratio;
        // Column tiles of the unit in channel-group order: q (Kt), k (Kt), v (Vb).
        auto pair_col = [&](uint32_t idx) -> uint32_t {
            if (idx < Kt) {
                return key_head * Kt + idx;
            }
            if (idx < 2 * Kt) {
                return Hk * Kt + key_head * Kt + (idx - Kt);
            }
            return 2 * Hk * Kt + head * Vt + v_first + (idx - 2 * Kt);
        };

        // ---- load_pair_block ----
        cb_reserve_back(cb_conv_w, conv_w_pages);
        cb_reserve_back(cb_gate_w, 2);
        cb_reserve_back(cb_norm_w, Vb);
        const uint32_t conv_w_base = get_write_ptr(cb_conv_w);
        for (uint32_t idx = 0; idx < pair_cols; ++idx) {
            const uint32_t ct = pair_col(idx);
            if constexpr (CONV_W_RAW) {
                // Tap j stays at row j; compute broadcasts it by row index.
                read_row_span(conv_w, ct, 0, GDN_CONV_TAPS, conv_w_base + idx * TILE_BYTES);
            } else {
                for (uint32_t tap = 0; tap < GDN_CONV_TAPS; ++tap) {
                    rows.read(conv_w, ct, tap, conv_w_base + (GDN_CONV_TAPS * idx + tap) * TILE_BYTES, 0);
                }
            }
        }
        // Rank-1 weights: compute broadcasts row 0.
        const uint32_t gate_w_base = get_write_ptr(cb_gate_w);
        read_row_span(a_log, head / TILE_ROWS, 0, 1, gate_w_base);
        read_row_span(dt_bias, head / TILE_ROWS, 0, 1, gate_w_base + TILE_BYTES);
        const uint32_t norm_w_base = get_write_ptr(cb_norm_w);
        for (uint32_t vt = 0; vt < Vb; ++vt) {
            read_row_span(norm_w, v_first + vt, 0, 1, norm_w_base + vt * TILE_BYTES);
        }
        // Completed by the first window group's flush (one DRAM round trip for weights, gates and window).
        bool weights_pending = true;

        const uint32_t seq_lo = seq * T;
        const uint32_t seq_hi = seq_lo + T;
        const uint32_t first_tr = seq_lo / TILE_ROWS;
        const uint32_t last_tr = (seq_hi - 1) / TILE_ROWS;
        for (uint32_t tr = first_tr; tr <= last_tr; ++tr) {
            const uint32_t base_row = tr * TILE_ROWS;
            const uint32_t lo = seq_lo > base_row ? seq_lo : base_row;
            const uint32_t hi = seq_hi < base_row + TILE_ROWS ? seq_hi : base_row + TILE_ROWS;

            // ---- gate inputs: token rows of the b, a tiles of the gate column ----
            cb_reserve_back(cb_gate_in, 2);
            const uint32_t gate_in_base = get_write_ptr(cb_gate_in);
            const uint32_t gate_page = tr * Gt + head / TILE_ROWS;
            read_row_span(b_acc, gate_page, lo - base_row, hi - base_row, gate_in_base);
            read_row_span(a_acc, gate_page, lo - base_row, hi - base_row, gate_in_base + TILE_BYTES);
            bool gates_pending = true;  // pushed after the first window group's flush

            // ---- conv window: X_j[p] = xx[row(p) + j - 3], WINDOW_GROUP_COLS column tiles per hand-off ----
            // Per group: one zero fill, one flush of every DRAM row read (current rows + history rows),
            // then one flush of the L1 row shifts that source the current-row tiles. A uniform group
            // size keeps every reservation contiguous in cb_window (its size is a multiple of a group).
            constexpr uint32_t group_cols = WINDOW_GROUP_COLS;
            constexpr uint32_t group_tiles = GDN_CONV_TAPS * group_cols;
            for (uint32_t group_first = 0; group_first < pair_cols; group_first += group_cols) {
                cb_reserve_back(cb_window, group_tiles);
                const uint32_t win = get_write_ptr(cb_window);
                if (zero_tokens != 0) {
                    // The writer zeroes the buffer slot by slot at kernel start, off this RISC's critical path.
                    cb_wait_front(cb_window_zeroed, 1);
                    cb_pop_front(cb_window_zeroed, 1);
                    --zero_tokens;
                }
                // Rows outside [lo, hi) are other sequences or padding: zero, never sourced.
                const uint32_t row_lo = lo - base_row;
                const uint32_t row_hi = hi - base_row;
                if (dirty_lo[window_slot] < row_lo || dirty_hi[window_slot] > row_hi) {
                    noc.async_write_zeros(window_buf, group_tiles * TILE_BYTES, {.offset_bytes = 0});
                    noc.write_zeros_l1_barrier();
                }
                dirty_lo[window_slot] = row_lo;
                dirty_hi[window_slot] = row_hi;
                window_slot = window_slot + 1 == WINDOW_SLOTS ? 0 : window_slot + 1;
                // History rows n = t + tap (t = row - seq_lo) all lie in conv_state rows
                // [HISTORY * seq, HISTORY * seq + HISTORY): read that span once per column into a staging
                // tile in place (one aligned read per face half) and copy rows out after the flush. A span
                // crossing a tile row is read row by row instead.
                const uint32_t hist_row0 = GDN_HISTORY * seq;
                const bool hist_needed = lo < seq_lo + GDN_HISTORY;
                const bool hist_staged = hist_needed && (hist_row0 % TILE_ROWS) + GDN_HISTORY <= TILE_ROWS;
                // The HISTORY token rows before lo, when all are rows of mixed_qkv inside one tile row (a tile
                // row continuing its sequence): read once per column as one aligned span into the same
                // staging tile, then copied out as row runs.
                const uint32_t prev_row0 = lo - GDN_HISTORY;
                const bool prev_staged = !hist_needed && prev_row0 / TILE_ROWS == (lo - 1) / TILE_ROWS;
                for (uint32_t col = 0; col < group_cols; ++col) {
                    const uint32_t ct = pair_col(group_first + col);
                    const uint32_t col_win = win + col * GDN_CONV_TAPS * TILE_BYTES;
                    read_tile_rows(
                        rows, mixed, tr * Ct + ct, base_row, lo, hi, col_win + (GDN_CONV_TAPS - 1) * TILE_BYTES);
                    if (hist_staged) {
                        const uint32_t r0 = hist_row0 % TILE_ROWS;
                        read_row_span(
                            conv_state,
                            (hist_row0 / TILE_ROWS) * Ct + ct,
                            r0,
                            r0 + GDN_HISTORY,
                            hist_stage + col * TILE_BYTES);
                    } else if (prev_staged) {
                        const uint32_t r0 = prev_row0 % TILE_ROWS;
                        read_row_span(
                            mixed,
                            (prev_row0 / TILE_ROWS) * Ct + ct,
                            r0,
                            r0 + GDN_HISTORY,
                            hist_stage + col * TILE_BYTES);
                        continue;
                    }
                    for (uint32_t tap = 0; tap < GDN_CONV_TAPS - 1; ++tap) {
                        const uint32_t shift = GDN_HISTORY - tap;  // source row = row - shift
                        const uint32_t remote_hi = hi < lo + shift ? hi : lo + shift;
                        for (uint32_t row = lo; row < remote_hi; ++row) {
                            const uint32_t p = row - base_row;
                            const uint32_t tap_tile = col_win + tap * TILE_BYTES;
                            if (row >= seq_lo + shift) {
                                const uint32_t src = row - shift;
                                rows.read(mixed, (src / TILE_ROWS) * Ct + ct, src % TILE_ROWS, tap_tile, p);
                            } else if (!hist_staged) {
                                const uint32_t src = hist_row0 + (row - seq_lo) + tap;
                                rows.read(conv_state, (src / TILE_ROWS) * Ct + ct, src % TILE_ROWS, tap_tile, p);
                            }
                        }
                    }
                }
                rows.finish();
                if (hist_staged || prev_staged) {
                    for (uint32_t col = 0; col < group_cols; ++col) {
                        const uint32_t col_win = win + col * GDN_CONV_TAPS * TILE_BYTES;
                        for (uint32_t tap = 0; tap < GDN_CONV_TAPS - 1; ++tap) {
                            const uint32_t shift = GDN_HISTORY - tap;
                            uint32_t end = hi < lo + shift ? hi : lo + shift;
                            uint32_t src = (prev_row0 + tap) % TILE_ROWS;  // row lo - shift
                            if (hist_staged) {
                                end = end < seq_lo + shift ? end : seq_lo + shift;
                                src = hist_row0 % TILE_ROWS + (lo - seq_lo) + tap;
                            }
                            if (end > lo) {
                                copy_local_rows(
                                    hist_stage + col * TILE_BYTES,
                                    src,
                                    col_win + tap * TILE_BYTES,
                                    lo - base_row,
                                    end - lo);
                            }
                        }
                    }
                }
                if (weights_pending) {
                    cb_push_back(cb_conv_w, conv_w_pages);
                    cb_push_back(cb_gate_w, 2);
                    cb_push_back(cb_norm_w, Vb);
                    weights_pending = false;
                }
                if (gates_pending) {
                    cb_push_back(cb_gate_in, 2);
                    gates_pending = false;
                }
                if (hi - lo > 1 || hist_staged || prev_staged) {
                    for (uint32_t col = 0; col < group_cols && hi - lo > 1; ++col) {
                        const uint32_t col_win = win + col * GDN_CONV_TAPS * TILE_BYTES;
                        const uint32_t cur_tap = col_win + (GDN_CONV_TAPS - 1) * TILE_BYTES;
                        for (uint32_t tap = 0; tap < GDN_CONV_TAPS - 1; ++tap) {
                            const uint32_t shift = GDN_HISTORY - tap;
                            if (hi > lo + shift) {
                                copy_local_rows(
                                    cur_tap,
                                    lo - base_row,
                                    col_win + tap * TILE_BYTES,
                                    lo + shift - base_row,
                                    hi - lo - shift);
                            }
                        }
                    }
                    noc_async_read_barrier();
                }
                cb_push_back(cb_window, group_tiles);
            }
            // ---- S0 of the piece (Kt x Vb slice of the pair state), after the first tile row's prep inputs ----
            if (tr == first_tr) {
                cb_reserve_back(cb_state_in, n_state);
                cb_reserve_back(cb_state_in_mm, n_state);
                const uint32_t state_base = get_write_ptr(cb_state_in);
                for (uint32_t i = 0; i < n_state; ++i) {
                    const uint32_t page = pair * Kt * Vt + (i / Vb) * Vt + v_first + i % Vb;
                    noc_async_read(state.get_noc_addr(page), state_base + i * STATE_TILE_BYTES, STATE_TILE_BYTES);
                }
                noc_async_read_barrier();
                cb_push_back(cb_state_in, n_state);
                cb_push_back(cb_state_in_mm, n_state);
            }

            // ---- z columns of the piece ----
            cb_reserve_back(cb_z, Vb);
            const uint32_t z_base = get_write_ptr(cb_z);
            for (uint32_t vt = 0; vt < Vb; ++vt) {
                const uint32_t page = tr * z_width_tiles + head * Vt + v_first + vt;
                read_tile_rows(rows, z_acc, page, base_row, lo, hi, z_base + vt * TILE_BYTES);
            }
            rows.finish();
            cb_push_back(cb_z, Vb);
        }

        // ---- copy_conv_state_block: conv_state_out[3b + n] = xx[b, T + n] ----
        // Columns are staged HISTORY rows apiece in the rows of one scratch tile: one read flush and one
        // write barrier per CS_COLS_PER_FLUSH columns.
        const uint32_t scratch = get_write_ptr(cb_scratch);
        // q/k columns are written by one unit per key head (first head, first piece).
        const uint32_t first_idx = (head % ratio == 0 && v_first == 0) ? 0 : 2 * Kt;
        for (uint32_t flush_first = first_idx; flush_first < pair_cols; flush_first += CS_COLS_PER_FLUSH) {
            const uint32_t flush_end =
                pair_cols < flush_first + CS_COLS_PER_FLUSH ? pair_cols : flush_first + CS_COLS_PER_FLUSH;
            for (uint32_t idx = flush_first; idx < flush_end; ++idx) {
                const uint32_t ct = pair_col(idx);
                const uint32_t slot_row = (idx - flush_first) * GDN_HISTORY;
                for (uint32_t n = 0; n < GDN_HISTORY; ++n) {
                    if (T + n < GDN_HISTORY) {
                        const uint32_t src = GDN_HISTORY * seq + T + n;
                        rows.read(conv_state, (src / TILE_ROWS) * Ct + ct, src % TILE_ROWS, scratch, slot_row + n);
                    } else {
                        const uint32_t src = seq_lo + T + n - GDN_HISTORY;
                        rows.read(mixed, (src / TILE_ROWS) * Ct + ct, src % TILE_ROWS, scratch, slot_row + n);
                    }
                }
            }
            rows.finish();
            for (uint32_t idx = flush_first; idx < flush_end; ++idx) {
                const uint32_t ct = pair_col(idx);
                const uint32_t slot_row = (idx - flush_first) * GDN_HISTORY;
                for (uint32_t n = 0; n < GDN_HISTORY; ++n) {
                    const uint32_t dst = GDN_HISTORY * seq + n;
                    write_row(conv_state_out, scratch, slot_row + n, (dst / TILE_ROWS) * Ct + ct, dst % TILE_ROWS);
                }
            }
            noc_async_write_barrier();
        }
    }
}
