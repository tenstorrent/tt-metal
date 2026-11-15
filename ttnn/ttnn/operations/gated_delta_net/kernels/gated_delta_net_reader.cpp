// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gated_delta_net reader (NCRISC).
//
// Per unit (pair (b, h), piece) of this core's contiguous range; a piece owns Vb = Vt / PIECES of the
// pair's v / state columns (R-VSPLIT; R-STEP is PIECES = 1) and every q/k column of its key head; under R-BCAST
// the key head's q injector reads only its q columns, the k injector only its k columns, a receiver none:
//   load_pair_block: conv_weight tiles (raw for the FPU conv, else row j -> row 0 of tap tile j), A_log / dt_bias
//                    gate-column tiles, norm_weight tiles.
//   per tile row of the pair: b/a gate tiles, the conv window (4 tap tiles per column tile, rows
//                    outside the pair's token rows zero), S0 after the first tile row's prep
//                    inputs (stall shadow), z tiles.
//   copy_conv_state_block: conv_state_out rows 3b..3b+2 of the pair's columns (bit-exact DM copy).
// Token rows are moved as face-row segments (64 B fp32, 32 B bf16; bf16 reads go through RowReader
// staging) unless the pair owns the whole tile row.
// R-PREP (ROLE): a prep core's unit is a (sequence, key head) with the r value heads' v columns, and it reads the
// gate, window and z inputs of tile rows offset, offset + stride, ... only (and norm_weight once); a state core reads
// S0, hands its inbox rows to compute as prep cores land them, and copies conv_state_out.

#include <stdint.h>

#include "api/alignment.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
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
constexpr uint32_t cb_in_kq = GDN_CB_IN_KQ;
constexpr uint32_t cb_in_knt = GDN_CB_IN_KNT;
constexpr uint32_t cb_in_mat = GDN_CB_IN_MAT;
constexpr uint32_t cb_in_vec = GDN_CB_IN_VEC;
constexpr uint32_t cb_in_v = GDN_CB_IN_V;

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
constexpr uint32_t ROLE = get_compile_time_arg_val(15);               // GDN_ROLE_*
constexpr uint32_t INBOX_SLOTS = get_compile_time_arg_val(16) > 0 ? get_compile_time_arg_val(16)
                                                                  : 1;  // R-PREP inbox slots (1 outside R-PREP)
constexpr bool PREP = ROLE == GDN_ROLE_PREP;    // R-PREP prep core: token-row inputs of strided tile rows
constexpr bool STATE = ROLE == GDN_ROLE_STATE;  // R-PREP state core: inbox, S0, z, norm_weight
constexpr bool QK_CORE = ROLE == GDN_ROLE_QK;   // R-QKCORE: q or k columns of one key head, every decode sequence
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
// are never consumed (weights read by row index, gate tiles read per token row): per 16-row band, one read
// from the band's rows in the left face through its rows in the right face, widened to DRAM read alignment
// and landed in place (one request instead of one per face, so no staging copy). Whole tile when every row
// is wanted. Completed by the next barrier / RowReader::finish().
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
        const uint32_t end = align_power_of_2(row_offset(hi - 1) + SEG_BYTES, READ_ALIGN) + FACE_BYTES;
        noc_async_read(acc.get_noc_addr(page, begin), dst + begin, end - begin);
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
    // v columns of a unit: a piece's, or on a prep core those of the key head's r value heads.
    constexpr uint32_t Vb = PREP ? ratio * Vt : Vt / PIECES;
    constexpr uint32_t n_state = Kt * Vb;
    constexpr uint32_t z_width_tiles = Hv * Vt;
    constexpr uint32_t pair_cols = 2 * Kt + Vb;

    constexpr auto mixed_args = TensorAccessorArgs<17>();
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
    const uint32_t qk_role = get_arg_val<uint32_t>(arg++);
    const uint32_t zero_slots = get_arg_val<uint32_t>(arg++);  // cb_window slots the writer zeroes
    // Tile rows of the unit's sequence this core handles: offset, offset + stride, ... (R-PREP prep core), or all.
    const uint32_t tr_offset = get_arg_val<uint32_t>(arg++);
    const uint32_t tr_stride = get_arg_val<uint32_t>(arg++);
    const uint32_t unit_stride = get_arg_val<uint32_t>(arg++);  // units first_unit, + stride, ... (R-QKCORE: Hv)
    // R-PREP state core: NoC (x, y) of its group's prep cores; tile row i of the sequence comes from prep core
    // i % producers into inbox slot i % INBOX_SLOTS.
    constexpr uint32_t MAX_PRODUCERS = STATE ? GDN_MAX_PREP_CORES_PER_GROUP : 1;
    uint32_t producers = 0;
    uint32_t producer_x[MAX_PRODUCERS];
    uint32_t producer_y[MAX_PRODUCERS];
    uint32_t granted_rows = 0;  // R-PREP state core: inbox rows granted to their prep cores
    if constexpr (STATE) {
        producers = get_arg_val<uint32_t>(arg++);
        for (uint32_t q = 0; q < producers; ++q) {
            producer_x[q] = get_arg_val<uint32_t>(arg++);
            producer_y[q] = get_arg_val<uint32_t>(arg++);
        }
    }
    // The unit's columns: q/k columns [qk_lo, qk_lo + qk_cols) of the pair's 2Kt, then its Vb v columns.
    const uint32_t qk_lo = qk_role == GDN_QK_K_INJECTOR ? Kt : 0;
    const uint32_t qk_cols = qk_role == GDN_QK_OWN ? 2 * Kt : (qk_role == GDN_QK_RECEIVER ? 0 : Kt);
    const uint32_t unit_cols = qk_cols + Vb;
    const uint32_t conv_w_pages = CONV_W_RAW ? unit_cols : GDN_CONV_TAPS * unit_cols;
    // R-VSPLIT hands the whole unit over at once; R-STEP has no receivers, so its group divides pair_cols.
    const uint32_t group_cols = WINDOW_GROUP_COLS == pair_cols ? unit_cols : WINDOW_GROUP_COLS;

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
    uint32_t zero_tokens = zero_slots;  // the writer zeroes slot k and then sends token k
    const uint32_t hist_stage = get_write_ptr(cb_hist_stage);
    const uint32_t cur_stage = hist_stage + group_cols * TILE_BYTES;  // single token rows, staged per column
    RowReader rows(STAGE_SLOTS == 0 ? 0 : get_write_ptr(cb_row_stage));

    dataflow_kernel_lib::
        prepare_reduce_scaler<cb_reduce_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(1.0f);
    if constexpr (CHUNKED && !STATE) {
        // Lower-triangular 0/1 mask (column <= row) of the chunk decay matrix, then the identity, once per kernel:
        // zeroed over the NoC, then only the ones are stored.
        CircularBuffer mask_buf(cb_tri_mask);
        cb_reserve_back(cb_tri_mask, 2);
        noc.async_write_zeros(mask_buf, 2 * STATE_TILE_BYTES, {.offset_bytes = 0});
        noc.write_zeros_l1_barrier();
        volatile tt_l1_ptr uint32_t* mask = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_tri_mask));
        constexpr uint32_t ONE_BITS = 0x3f800000u;
        constexpr uint32_t TILE_WORDS = TILE_ROWS * TILE_ROWS;
        for (uint32_t r = 0; r < TILE_ROWS; ++r) {
            const uint32_t row = (r >> 4) * 2 * 256 + (r & 15) * 16;
            for (uint32_t c = 0; c <= r; ++c) {
                mask[row + (c >> 4) * 256 + (c & 15)] = ONE_BITS;
            }
            mask[TILE_WORDS + row + (r >> 4) * 256 + (r & 15)] = ONE_BITS;  // identity (R-PREP inverse)
        }
        cb_push_back(cb_tri_mask, 2);
    }

    if constexpr (QK_CORE) {
        // ---- R-QKCORE: the q (or k) columns of key head first_unit for token rows [0, num_units) of the decode tile
        // row, row p being sequence p. Current rows and conv_state tiles are read whole into staging, then the rows
        // are copied into the zeroed window: tap 3 row p = x[p], tap j < 3 row p = conv_state[3p + j].
        const uint32_t batch = num_units;
        const uint32_t col0 = (qk_role == GDN_QK_K_INJECTOR ? Hk * Kt : 0) + first_unit * Kt;
        constexpr uint32_t w_pages = CONV_W_RAW ? Kt : GDN_CONV_TAPS * Kt;
        const uint32_t cs_tiles = (GDN_HISTORY * batch + TILE_ROWS - 1) / TILE_ROWS;
        const uint32_t cs_stage = hist_stage + Kt * TILE_BYTES;
        {
            DeviceZoneScopedN("QK-RD");
            cb_reserve_back(cb_conv_w, w_pages);
            const uint32_t conv_w_base = get_write_ptr(cb_conv_w);
            for (uint32_t col = 0; col < Kt; ++col) {
                if constexpr (CONV_W_RAW) {
                    read_row_span(conv_w, col0 + col, 0, GDN_CONV_TAPS, conv_w_base + col * TILE_BYTES);
                } else {
                    for (uint32_t tap = 0; tap < GDN_CONV_TAPS; ++tap) {
                        rows.read(conv_w, col0 + col, tap, conv_w_base + (GDN_CONV_TAPS * col + tap) * TILE_BYTES, 0);
                    }
                }
                noc_async_read(mixed.get_noc_addr(col0 + col), hist_stage + col * TILE_BYTES, TILE_BYTES);
                for (uint32_t i = 0; i < cs_tiles; ++i) {
                    noc_async_read(
                        conv_state.get_noc_addr(i * Ct + col0 + col),
                        cs_stage + (col * cs_tiles + i) * TILE_BYTES,
                        TILE_BYTES);
                }
            }
            rows.finish();
        }
        DeviceZoneScopedN("QK-COPY");
        cb_push_back(cb_conv_w, w_pages);
        // One window group per column (the writer zeroed one slot per column), so conv overlaps the next copies.
        for (uint32_t col = 0; col < Kt; ++col) {
            cb_reserve_back(cb_window, GDN_CONV_TAPS);
            cb_wait_front(cb_window_zeroed, 1);
            cb_pop_front(cb_window_zeroed, 1);
            const uint32_t col_win = get_write_ptr(cb_window);
            copy_local_rows(hist_stage + col * TILE_BYTES, 0, col_win + GDN_HISTORY * TILE_BYTES, 0, batch);
            for (uint32_t p = 0; p < batch; ++p) {
                for (uint32_t tap = 0; tap < GDN_HISTORY; ++tap) {
                    const uint32_t r = GDN_HISTORY * p + tap;
                    copy_local_rows(
                        cs_stage + (col * cs_tiles + r / TILE_ROWS) * TILE_BYTES,
                        r % TILE_ROWS,
                        col_win + tap * TILE_BYTES,
                        p,
                        1);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_window, GDN_CONV_TAPS);
        }
        return;
    }

    for (uint32_t pass = 0; pass < 2; ++pass) {
        for (uint32_t unit_idx = 0; unit_idx < num_units; ++unit_idx) {
            const uint32_t unit = first_unit + unit_idx * unit_stride;
            const uint32_t pair = unit / PIECES;
            const uint32_t v_first = PREP ? 0 : (unit % PIECES) * Vb;  // first v / state column tile of the piece
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

            // Pair column (channel-group order) of the unit's local column `local`.
            auto unit_col = [&](uint32_t local) -> uint32_t {
                return local < qk_cols ? qk_lo + local : 2 * Kt + (local - qk_cols);
            };

            // Pass 0 reads every unit's inputs; pass 1 copies conv_state_out, so a core's next unit is not read behind
            // this unit's copy.
            if (pass == 0) {
                // ---- load_pair_block ----
                // norm_weight columns of the unit (a prep core's r heads share the Vt tiles).
                constexpr uint32_t norm_w_pages = PREP ? Vt : Vb;
                if constexpr (!STATE) {
                    cb_reserve_back(cb_norm_w, norm_w_pages);
                    const uint32_t norm_w_base = get_write_ptr(cb_norm_w);
                    for (uint32_t vt = 0; vt < norm_w_pages; ++vt) {
                        read_row_span(norm_w, v_first + vt, 0, 1, norm_w_base + vt * TILE_BYTES);
                    }
                }
                if constexpr (!STATE) {
                    cb_reserve_back(cb_conv_w, conv_w_pages);
                    cb_reserve_back(cb_gate_w, 2);
                    const uint32_t conv_w_base = get_write_ptr(cb_conv_w);
                    for (uint32_t local = 0; local < unit_cols; ++local) {
                        const uint32_t ct = pair_col(unit_col(local));
                        if constexpr (CONV_W_RAW) {
                            // Tap j stays at row j; compute broadcasts it by row index.
                            read_row_span(conv_w, ct, 0, GDN_CONV_TAPS, conv_w_base + local * TILE_BYTES);
                        } else {
                            for (uint32_t tap = 0; tap < GDN_CONV_TAPS; ++tap) {
                                rows.read(conv_w, ct, tap, conv_w_base + (GDN_CONV_TAPS * local + tap) * TILE_BYTES, 0);
                            }
                        }
                    }
                    // Rank-1 weights: compute broadcasts row 0.
                    const uint32_t gate_w_base = get_write_ptr(cb_gate_w);
                    read_row_span(a_log, head / TILE_ROWS, 0, 1, gate_w_base);
                    read_row_span(dt_bias, head / TILE_ROWS, 0, 1, gate_w_base + TILE_BYTES);
                }
                // Completed by the first window group's flush (one DRAM round trip for weights, gates and window).
                bool weights_pending = !STATE;

                const uint32_t seq_lo = seq * T;
                const uint32_t seq_hi = seq_lo + T;
                const uint32_t first_tr = seq_lo / TILE_ROWS;
                const uint32_t last_tr = (seq_hi - 1) / TILE_ROWS;
                for (uint32_t tr = first_tr + tr_offset; tr <= last_tr; tr += tr_stride) {
                    const uint32_t base_row = tr * TILE_ROWS;
                    const uint32_t lo = seq_lo > base_row ? seq_lo : base_row;
                    const uint32_t hi = seq_hi < base_row + TILE_ROWS ? seq_hi : base_row + TILE_ROWS;

                    if constexpr (STATE) {
                        // ---- R-PREP inbox: grant each tile row's slot to its prep core as soon as compute has freed
                        // it (up to INBOX_SLOTS rows ahead, so the prep cores' writes overlap the chunk steps), then
                        // hand row i to compute once it has landed. Compute pops cb_in_v last of the inbox buffers, so
                        // a free cb_in_v slot frees the row's.
                        const uint32_t i = tr - first_tr;
                        const uint32_t n_rows = last_tr - first_tr + 1;
                        volatile tt_l1_ptr uint32_t* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            get_semaphore(GDN_LANDED_FIRST_SEM_ID + i % INBOX_SLOTS));
                        Semaphore<> granted(GDN_LANDED_FIRST_SEM_ID + INBOX_SLOTS);
                        auto grant_free_slots = [&]() {
                            while (granted_rows < n_rows &&
                                   cb_pages_reservable_at_back(cb_in_v, (granted_rows - i + 1) * Vb)) {
                                granted.up(
                                    noc, producer_x[granted_rows % producers], producer_y[granted_rows % producers], 1);
                                ++granted_rows;
                            }
                        };
                        if (granted_rows == i) {
                            cb_reserve_back(cb_in_v, Vb);  // row i's slot: freed once compute pops row i - INBOX_SLOTS
                        }
                        do {
                            grant_free_slots();
                        } while (*landed < i / INBOX_SLOTS + 1);
                        cb_reserve_back(cb_in_kq, 2 * Kt);
                        cb_reserve_back(cb_in_knt, Kt);
                        cb_reserve_back(cb_in_mat, GDN_IN_MAT_PAGES);
                        cb_reserve_back(cb_in_vec, GDN_IN_VEC_PAGES);
                        cb_reserve_back(cb_in_v, Vb);
                        cb_push_back(cb_in_kq, 2 * Kt);
                        cb_push_back(cb_in_knt, Kt);
                        cb_push_back(cb_in_mat, GDN_IN_MAT_PAGES);
                        cb_push_back(cb_in_vec, GDN_IN_VEC_PAGES);
                        cb_push_back(cb_in_v, Vb);
                    } else {
                        // ---- gate inputs: token rows of the b, a tiles of the gate column ----
                        cb_reserve_back(cb_gate_in, 2);
                        const uint32_t gate_in_base = get_write_ptr(cb_gate_in);
                        const uint32_t gate_page = tr * Gt + head / TILE_ROWS;
                        read_row_span(b_acc, gate_page, lo - base_row, hi - base_row, gate_in_base);
                        read_row_span(a_acc, gate_page, lo - base_row, hi - base_row, gate_in_base + TILE_BYTES);
                        bool gates_pending = true;  // pushed after the first window group's flush

                        // ---- conv window: X_j[p] = xx[row(p) + j - 3], WINDOW_GROUP_COLS column tiles per hand-off
                        // ---- Per group: one zero fill, one flush of every DRAM row read (current rows + history
                        // rows), then one flush of the L1 row shifts that source the current-row tiles. A uniform group
                        // size keeps every reservation contiguous in cb_window (its size is a multiple of a group).
                        const uint32_t group_tiles = GDN_CONV_TAPS * group_cols;
                        DeviceZoneScopedN("RD-WIN");
                        for (uint32_t group_first = 0; group_first < unit_cols; group_first += group_cols) {
                            cb_reserve_back(cb_window, group_tiles);
                            const uint32_t win = get_write_ptr(cb_window);
                            if (zero_tokens != 0) {
                                // The writer zeroes the buffer slot by slot at kernel start, off this RISC's critical
                                // path.
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
                            // The HISTORY token rows before lo, when all are rows of mixed_qkv inside one tile row (a
                            // tile row continuing its sequence): read once per column as one aligned span into the same
                            // staging tile, then copied out as row runs.
                            const uint32_t prev_row0 = lo - GDN_HISTORY;
                            const bool prev_staged = !hist_needed && prev_row0 / TILE_ROWS == (lo - 1) / TILE_ROWS;
                            // A single token row (decode): one request per column into staging, then copied into the
                            // tap tile, whose other rows must stay zero.
                            const bool cur_staged = hi - lo == 1;
                            for (uint32_t col = 0; col < group_cols; ++col) {
                                const uint32_t ct = pair_col(unit_col(group_first + col));
                                const uint32_t col_win = win + col * GDN_CONV_TAPS * TILE_BYTES;
                                if (cur_staged) {
                                    read_row_span(
                                        mixed,
                                        tr * Ct + ct,
                                        lo - base_row,
                                        hi - base_row,
                                        cur_stage + col * TILE_BYTES);
                                } else {
                                    read_tile_rows(
                                        rows,
                                        mixed,
                                        tr * Ct + ct,
                                        base_row,
                                        lo,
                                        hi,
                                        col_win + (GDN_CONV_TAPS - 1) * TILE_BYTES);
                                }
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
                                            rows.read(
                                                conv_state, (src / TILE_ROWS) * Ct + ct, src % TILE_ROWS, tap_tile, p);
                                        }
                                    }
                                }
                            }
                            rows.finish();
                            if (hist_staged || prev_staged || cur_staged) {
                                for (uint32_t col = 0; col < group_cols; ++col) {
                                    const uint32_t col_win = win + col * GDN_CONV_TAPS * TILE_BYTES;
                                    if (cur_staged) {
                                        copy_local_rows(
                                            cur_stage + col * TILE_BYTES,
                                            lo - base_row,
                                            col_win + (GDN_CONV_TAPS - 1) * TILE_BYTES,
                                            lo - base_row,
                                            1);
                                    }
                                    for (uint32_t tap = 0; tap < GDN_CONV_TAPS - 1 && (hist_staged || prev_staged);
                                         ++tap) {
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
                                cb_push_back(cb_norm_w, norm_w_pages);
                                weights_pending = false;
                            }
                            if (gates_pending) {
                                cb_push_back(cb_gate_in, 2);
                                gates_pending = false;
                            }
                            if (hi - lo > 1 || hist_staged || prev_staged || cur_staged) {
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
                    }
                    // ---- S0 of the piece (Kt x Vb slice of the pair state), after the first tile row's prep inputs
                    // ----
                    if (tr == first_tr && !PREP) {
                        DeviceZoneScopedN("RD-S0");
                        cb_reserve_back(cb_state_in, n_state);
                        cb_reserve_back(cb_state_in_mm, n_state);
                        const uint32_t state_base = get_write_ptr(cb_state_in);
                        for (uint32_t i = 0; i < n_state; ++i) {
                            const uint32_t page = pair * Kt * Vt + (i / Vb) * Vt + v_first + i % Vb;
                            noc_async_read(
                                state.get_noc_addr(page), state_base + i * STATE_TILE_BYTES, STATE_TILE_BYTES);
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_state_in, n_state);
                        cb_push_back(cb_state_in_mm, n_state);
                    }

                    if constexpr (STATE) {
                        continue;
                    }
                    // ---- z columns of the unit ----
                    cb_reserve_back(cb_z, Vb);
                    const uint32_t z_base = get_write_ptr(cb_z);
                    for (uint32_t vt = 0; vt < Vb; ++vt) {
                        const uint32_t page = tr * z_width_tiles + head * Vt + v_first + vt;
                        read_tile_rows(rows, z_acc, page, base_row, lo, hi, z_base + vt * TILE_BYTES);
                    }
                    rows.finish();
                    cb_push_back(cb_z, Vb);
                }

                continue;
            }
            if constexpr (PREP) {
                continue;
            }
            // ---- copy_conv_state_block: conv_state_out[3b + n] = xx[b, T + n] ----
            // Columns are staged HISTORY rows apiece in the rows of one scratch tile: one read flush and one
            // write barrier per CS_COLS_PER_FLUSH columns.
            DeviceZoneScopedN("RD-CS");
            const uint32_t scratch = get_write_ptr(cb_scratch);
            const uint32_t seq_lo = seq * T;
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
}
