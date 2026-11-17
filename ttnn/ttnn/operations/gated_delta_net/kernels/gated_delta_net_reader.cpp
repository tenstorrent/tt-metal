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
// R-PREP (ROLE): a prep core's unit is a (sequence, key head) with PREP_HEADS of its value heads' v columns, and it
// reads the gate, window and z inputs of tile rows offset, offset + stride, ... only (and norm_weight once); a state
// core reads S0, hands its inbox rows to compute as prep cores land them, and copies conv_state_out.

#include <stdint.h>

#include "api/alignment.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "gated_delta_net_qk_rows.hpp"

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
constexpr uint32_t cb_in_kq = GDN_CB_IN_KQ;
constexpr uint32_t cb_in_knt = GDN_CB_IN_KNT;
constexpr uint32_t cb_in_mat = GDN_CB_IN_MAT;
constexpr uint32_t cb_in_vec = GDN_CB_IN_VEC;
constexpr uint32_t cb_in_v = GDN_CB_IN_V;
constexpr uint32_t cb_kq_rows = GDN_CB_KQ_ROWS;

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
constexpr uint32_t S0_READ_TILES = get_compile_time_arg_val(17);        // S0 tiles per DRAM read barrier
// conv_state_out writers (host cs_split): 0 pair cores (q / k columns by one unit per key head), 1 the q / k cores
// alone, 2 the q / k cores their own columns and the pair cores their v columns.
constexpr uint32_t CS_SPLIT = get_compile_time_arg_val(18);
constexpr bool CS_BY_QK_CORES = CS_SPLIT == 1;
constexpr bool DECODE = get_compile_time_arg_val(19) != 0;  // T = 1 (other builds compile no prefetch path)
constexpr uint32_t PREP_HEADS = get_compile_time_arg_val(20);  // R-PREP: value heads per prep group
constexpr bool PREP = ROLE == GDN_ROLE_PREP;    // R-PREP prep core: token-row inputs of strided tile rows
constexpr bool STATE = ROLE == GDN_ROLE_STATE;  // R-PREP state core: inbox, S0, z, norm_weight
constexpr bool QK_CORE = ROLE == GDN_ROLE_QK;   // R-QKCORE: q or k columns of one key head, every decode sequence
constexpr bool PREP_QK = ROLE == GDN_ROLE_PREP_QK;  // R-PREP q / k core: window of its q or k columns, nothing else
constexpr bool NORM_SPLIT = PREP && GDN_PREP_NORM_SPLIT != 0;  // z of the rows it normalizes, after its prep rows
// Host SEQ_SPLIT: the sequence in SEGS segments (sequences 0 .. SEGS - 1 of the plan); sequences SEGS.. are the P
// halves of segments 1.. (state cores only). A state core of a segment 1.. starts from 0 (its P half from I), takes P
// and the entering state in cb_seg_in once they have landed (SEG_SEM_ID counts them) and then the tile rows of Oh_P
// from the P cores of its head (credits to them, landings counted per P core on SEG_SEM_ID + 1 + piece).
constexpr uint32_t SEGS = GDN_SEQ_SPLIT;
constexpr bool SPLIT = SEGS > 1;
constexpr uint32_t cb_seg_in = GDN_CB_SEG_IN;
constexpr uint32_t cb_seg_row = GDN_CB_SEG_ROW;
constexpr uint32_t SEG_SEM_ID = GDN_LANDED_FIRST_SEM_ID + INBOX_SLOTS;
// Host STATE_NORM_TAIL: tile rows per sequence whose RMS norm the state cores run (z and norm_weight read here).
constexpr uint32_t TAIL = GDN_NORM_TAIL;
// PREP_SHARE_QK semaphores after the inbox's: qn / kn tile rows landed on a prep core (cumulative each).
constexpr uint32_t QN_LANDED_SEM_ID = GDN_LANDED_FIRST_SEM_ID + INBOX_SLOTS + 2;
constexpr uint32_t KN_LANDED_SEM_ID = QN_LANDED_SEM_ID + 1;
constexpr uint32_t KQ_GRANT_FIRST_SEM_ID = KN_LANDED_SEM_ID + 1;    // on a q / k core, one per destination prep core
constexpr uint32_t FACE_BYTES = TILE_BYTES / 4;                     // one 16x16 face
constexpr uint32_t SEG_BYTES = FACE_BYTES / 16;                     // one face row
constexpr uint32_t CS_COLS_PER_FLUSH = TILE_ROWS / GDN_HISTORY;     // conv_state_out columns per scratch tile
// Decode NoC read transaction ids (trid 0 also counts other requests of the NIU): the first window group's staged rows,
// then the unit's other inputs issued before that group's flush.
constexpr uint32_t WINDOW_TRID = 1;
constexpr uint32_t INPUTS_TRID = 2;
constexpr uint32_t S0_TRID = 3;  // decode: a unit's S0, issued with its window rows, awaited where compute needs it
constexpr bool EARLY_S0 = GDN_EARLY_S0 != 0;
// conv_state_out is conv_state: a core overwrites conv_state rows / columns other cores read only after their reads
// have landed (each reader signals the writer's GDN_CS_READ_SEM_ID).
constexpr bool CS_IN_PLACE = GDN_CS_IN_PLACE != 0;
// R-QKCORE pair cores: this reader writes its share of the final-state tiles on its NoC (runtime reader_share, see
// gdn::state_tile_on_reader), the writer the rest on the other one. It never pops cb_state_out: it follows compute's
// pushes by the received count and signals the writer, which pops a state only after both parts have left
// (cb_state_done).
constexpr bool STATE_SPLIT = GDN_STATE_SPLIT != 0;
constexpr uint32_t cb_state_out = GDN_CB_STATE_OUT;
constexpr uint32_t cb_state_done = GDN_CB_STATE_DONE;
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

    // Completes staged rows only if any are pending (a full barrier then); else nothing.
    void finish_staged() {
        if (pending_ != 0) {
            finish();
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
    const gdn::NocDrainOnReturn drain;
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t Hv = get_compile_time_arg_val(2);
    constexpr uint32_t Hk = get_compile_time_arg_val(3);
    constexpr uint32_t Ct = get_compile_time_arg_val(4);
    constexpr uint32_t Gt = get_compile_time_arg_val(5);
    constexpr uint32_t ratio = Hv / Hk;
    // v columns of a unit: a piece's, or on a prep core those of its group's PREP_HEADS value heads.
    constexpr uint32_t Vb = PREP ? PREP_HEADS * Vt : Vt / PIECES;
    constexpr uint32_t n_state = Kt * Vb;
    constexpr uint32_t z_width_tiles = Hv * Vt;
    constexpr uint32_t pair_cols = 2 * Kt + Vb;
    // Row width in tiles and first tile column of mixed_qkv, z and the gates. Packed form: z_acc, b_acc and a_acc are
    // mixed_qkv's accessor, z follows the Ct conv columns and the gate tile (a, then b) follows z.
    constexpr bool PACKED = GDN_PACKED_TILES != 0;
    constexpr uint32_t Mt = PACKED ? GDN_PACKED_TILES : Ct;
    constexpr uint32_t z_row_tiles = PACKED ? Mt : z_width_tiles;
    constexpr uint32_t z_col0 = PACKED ? Ct : 0;
    constexpr uint32_t gate_row_tiles = PACKED ? Mt : Gt;
    constexpr uint32_t gate_col0 = PACKED ? GDN_PACKED_GATE_TILE : 0;
    static_assert(!PACKED || 2 * Hv <= TILE_ROWS, "packed gates: a and b in one tile");

    constexpr auto mixed_args = TensorAccessorArgs<21>();
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
    constexpr auto state_out_args = TensorAccessorArgs<conv_state_out_args.next_compile_time_args_offset()>();

    // Buffer addresses are common runtime args: a cached program patches them per call without per-core args.
    uint32_t addr = 0;
    const auto mixed = TensorAccessor(mixed_args, get_common_arg_val<uint32_t>(addr++), TILE_BYTES);
    const auto z_acc = TensorAccessor(z_args, get_common_arg_val<uint32_t>(addr++), TILE_BYTES);
    const auto b_acc = TensorAccessor(b_args, get_common_arg_val<uint32_t>(addr++), TILE_BYTES);
    const auto a_acc = TensorAccessor(a_args, get_common_arg_val<uint32_t>(addr++), TILE_BYTES);
    const auto conv_state = TensorAccessor(conv_state_args, get_common_arg_val<uint32_t>(addr++), TILE_BYTES);
    const auto state = TensorAccessor(state_args, get_common_arg_val<uint32_t>(addr++), STATE_TILE_BYTES);
    const auto conv_w = TensorAccessor(conv_w_args, get_common_arg_val<uint32_t>(addr++), TILE_BYTES);
    const auto a_log = TensorAccessor(a_log_args, get_common_arg_val<uint32_t>(addr++), TILE_BYTES);
    const auto dt_bias = TensorAccessor(dt_bias_args, get_common_arg_val<uint32_t>(addr++), TILE_BYTES);
    const auto norm_w = TensorAccessor(norm_w_args, get_common_arg_val<uint32_t>(addr++), TILE_BYTES);
    const auto conv_state_out = TensorAccessor(conv_state_out_args, get_common_arg_val<uint32_t>(addr++), TILE_BYTES);
    const auto state_out = TensorAccessor(state_out_args, get_common_arg_val<uint32_t>(addr++), STATE_TILE_BYTES);
    // Compute pushes no final state before this reader has supplied S0: the count now is the base of every state's.
    const uint16_t state_out_pushed0 = STATE_SPLIT ? reg_read((uintptr_t)get_cb_tiles_received_ptr(cb_state_out)) : 0;
    uint32_t arg = 0;
    const uint32_t first_unit = get_arg_val<uint32_t>(arg++);
    const uint32_t num_units = get_arg_val<uint32_t>(arg++);
    const uint32_t T = get_arg_val<uint32_t>(arg++);
    // Compile-time when every core of the kernel group shares it (descriptor GDN_GROUP_QK_ROLE).
    const uint32_t qk_role =
        GDN_GROUP_QK_ROLE == GDN_QK_PER_CORE ? get_arg_val<uint32_t>(arg) : uint32_t{GDN_GROUP_QK_ROLE};
    ++arg;
    const uint32_t zero_slots = get_arg_val<uint32_t>(arg++);  // cb_window slots the writer zeroes
    // Tile rows of the unit's sequence this core handles: offset, offset + stride, ... (R-PREP prep core), or all.
    const uint32_t tr_offset = get_arg_val<uint32_t>(arg++);
    const uint32_t tr_stride = get_arg_val<uint32_t>(arg++);
    const uint32_t unit_stride = get_arg_val<uint32_t>(arg++);  // units first_unit, + stride, ... (R-QKCORE: Hv)
    // Host PREP_START_STAGGER (cycles): the DRAM banks serve the first reads of every core at once, and only tile
    // row 0's set when the chain starts: the prep cores of the other stride positions issue theirs that much later,
    // the state cores (S0 is read long before row 0 lands) twice that.
    if constexpr (GDN_START_STAGGER != 0 || GDN_START_STATE != 0) {
        if ((PREP || PREP_QK) && tr_offset != 0) {
            riscv_wait(GDN_START_STAGGER);
        } else if (STATE) {
            riscv_wait(GDN_START_STATE);
        }
    }
    // R-PREP prep core, NORM_SPLIT: the norm units (tile row i, value head j of the group: unit i * PREP_HEADS + j)
    // it normalizes: offset, + stride, ...
    const uint32_t norm_offset = PREP ? get_arg_val<uint32_t>(arg++) : 0;
    const uint32_t norm_stride = PREP ? get_arg_val<uint32_t>(arg++) : 1;
    const uint32_t norm_mid = PREP ? get_arg_val<uint32_t>(arg++) : 0;  // of them, ahead of each later row
    // R-PREP prep core under PREP_SHARE_QK (qk role receiver): NoC (x, y) of its position's q and k cores, then its
    // index among their destinations (its grant counter there).
    const bool kq_landing = PREP && qk_role == GDN_QK_RECEIVER;
    uint32_t kq_src[4] = {0, 0, 0, 0};
    uint32_t kq_grant_sem = 0;
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
    uint32_t seg_peer_x[PIECES];
    uint32_t seg_peer_y[PIECES];
    if constexpr (STATE && SPLIT) {
        for (uint32_t p = 0; p < PIECES; ++p) {
            seg_peer_x[p] = get_arg_val<uint32_t>(arg++);
            seg_peer_y[p] = get_arg_val<uint32_t>(arg++);
        }
        arg += 2;  // the next segment's piece (writer)
    }
    if (kq_landing) {
        for (uint32_t i = 0; i < 4; ++i) {
            kq_src[i] = get_arg_val<uint32_t>(arg++);
        }
        kq_grant_sem = KQ_GRANT_FIRST_SEM_ID + get_arg_val<uint32_t>(arg++);
    }
    // The unit's columns: q/k columns [qk_lo, qk_lo + qk_cols) of the pair's 2Kt, then its Vb v columns (none on an
    // R-PREP q / k core).
    const uint32_t qk_lo = qk_role == GDN_QK_K_INJECTOR ? Kt : 0;
    const uint32_t qk_cols = qk_role == GDN_QK_OWN ? 2 * Kt : (qk_role == GDN_QK_RECEIVER ? 0 : Kt);
    const uint32_t unit_cols = qk_cols + (PREP_QK ? 0 : Vb);
    const uint32_t conv_w_pages = CONV_W_RAW ? unit_cols : GDN_CONV_TAPS * unit_cols;
    // R-VSPLIT hands the whole unit over at once; R-STEP has no receivers, so its group divides pair_cols.
    const uint32_t group_cols = WINDOW_GROUP_COLS == pair_cols ? unit_cols : WINDOW_GROUP_COLS;

    // The piece's Kt x Vb slice of pair `pair`'s S0 into `dst`; the caller barriers.
    // Tiles from (pair * STATE_STAGGER) mod n_state on, so that cores starting together spread over the banks.
    auto read_s0 = [&](uint32_t dst, uint32_t pair, uint32_t v_first) {
        gdn::read_state_slice<Kt, Vt, Vb, STATE_TILE_BYTES>(
            state, dst, pair, v_first, GDN_STATE_STAGGER, S0_READ_TILES);
    };

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

    if constexpr (QK_CORE) {
        // ---- R-QKCORE: the q (or k) columns of key head first_unit for token rows [0, num_units) of the decode tile
        // row, row p being sequence p. Current rows and conv_state tiles are read whole into staging, then the rows
        // are copied into the zeroed window: tap 3 row p = x[p], tap j < 3 row p = conv_state[3p + j]. The writer
        // zeroes each column's slot and copies the history rows from history_split(B) on (after the staging token on
        // cb_scratch), then hands the slot over (cb_window_zeroed); this RISC copies the rest and pushes it.
        GDN_ZONE("gdn_rd_qk");
        const uint32_t batch = num_units;
        const uint32_t hist_pct = qk_role == GDN_QK_Q_INJECTOR ? GDN_QK_Q_READER_PCT : 50;
        const uint32_t col0 = (qk_role == GDN_QK_K_INJECTOR ? Hk * Kt : 0) + first_unit * Kt;
        constexpr uint32_t w_pages = CONV_W_RAW ? Kt : GDN_CONV_TAPS * Kt;
        const uint32_t cs_tiles = (GDN_HISTORY * batch + TILE_ROWS - 1) / TILE_ROWS;
        const uint32_t cs_stage = hist_stage + Kt * TILE_BYTES;
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
        GDN_MARK("m_rd_inputs");
        cb_reserve_back(cb_scratch, 1);
        cb_push_back(cb_scratch, 1);
        cb_push_back(cb_conv_w, w_pages);
        // One window group per column (the writer's slot token: zeroed, its rows copied), so conv overlaps the next
        // column's copies.
        for (uint32_t col = 0; col < Kt; ++col) {
            cb_reserve_back(cb_window, GDN_CONV_TAPS);
            cb_wait_front(cb_window_zeroed, 1);
            cb_pop_front(cb_window_zeroed, 1);
            const uint32_t col_win = get_write_ptr(cb_window);
            copy_local_rows(hist_stage + col * TILE_BYTES, 0, col_win + GDN_HISTORY * TILE_BYTES, 0, batch);
            gdn::copy_history_rows<TILE_BYTES, GDN_HISTORY>(
                cs_stage + col * cs_tiles * TILE_BYTES, col_win, 0, gdn::history_split(batch, hist_pct));
            noc_async_read_barrier();
            cb_push_back(cb_window, GDN_CONV_TAPS);
            GDN_MARK("m_rd_col");
        }

        // ---- conv_state_out of the key head's columns, whole tiles (the writer's copies are done: every slot token
        // has been taken, so the staging is free): rows 3b + n = conv_state rows 3b + n + 1 for n < 2, row 3b + 2 =
        // x[b]. Staging: the x tile, the conv_state tiles, then the output tiles. First this core's own columns, then
        // its share of the v columns, which in place wait until the key head's pair cores have read their history rows.
        const uint32_t own_lo = get_arg_val<uint32_t>(arg++);
        const uint32_t own_hi = get_arg_val<uint32_t>(arg++);
        const uint32_t v_lo = get_arg_val<uint32_t>(arg++);
        const uint32_t v_hi = get_arg_val<uint32_t>(arg++);
        const uint32_t cs_reads = get_arg_val<uint32_t>(arg++);
        const uint32_t x_tile = hist_stage;
        const uint32_t cs_in = x_tile + TILE_BYTES;
        const uint32_t cs_out = cs_in + cs_tiles * TILE_BYTES;
        const uint32_t out_rows = GDN_HISTORY * batch;
        for (uint32_t idx = own_lo; idx < own_hi + (v_hi - v_lo); ++idx) {
            if (CS_IN_PLACE && idx == own_hi) {
                Semaphore<>(GDN_CS_READ_SEM_ID).wait_min(cs_reads);
            }
            const uint32_t ct = idx < own_hi ? idx : v_lo + (idx - own_hi);
            noc_async_read(mixed.get_noc_addr(ct), x_tile, TILE_BYTES);
            for (uint32_t i = 0; i < cs_tiles; ++i) {
                noc_async_read(conv_state.get_noc_addr(i * Ct + ct), cs_in + i * TILE_BYTES, TILE_BYTES);
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < cs_tiles; ++i) {
                // Output rows [32 i, 32 i + n) <- conv_state rows one below: within the tile, then its last row from
                // the next tile's first.
                const uint32_t rows_here = out_rows - i * TILE_ROWS < TILE_ROWS ? out_rows - i * TILE_ROWS : TILE_ROWS;
                const uint32_t shifted = rows_here < TILE_ROWS - 1 ? rows_here : TILE_ROWS - 1;
                copy_local_rows(cs_in + i * TILE_BYTES, 1, cs_out + i * TILE_BYTES, 0, shifted);
                if (rows_here == TILE_ROWS && i + 1 < cs_tiles) {
                    copy_local_rows(cs_in + (i + 1) * TILE_BYTES, 0, cs_out + i * TILE_BYTES, TILE_ROWS - 1, 1);
                }
            }
            noc_async_read_barrier();
            for (uint32_t b = 0; b < batch; ++b) {
                const uint32_t r = GDN_HISTORY * b + GDN_HISTORY - 1;
                copy_local_rows(x_tile, b, cs_out + (r / TILE_ROWS) * TILE_BYTES, r % TILE_ROWS, 1);
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < cs_tiles; ++i) {
                noc_async_write(cs_out + i * TILE_BYTES, conv_state_out.get_noc_addr(i * Ct + ct), TILE_BYTES);
            }
            noc_async_write_barrier();
        }
        return;
    }

    // A core's later units of one value head (R-QKCORE: consecutive sequences) copy the head's weights and the decode
    // gate tiles from the previous unit's slots instead of reading them from DRAM again, where page 0 of every small
    // tensor sits in one bank. The previous slots are intact: a buffer is reused two units later at the earliest (or,
    // at depth 1, is the same slot).
    // In place: after its unit cs_signal_unit's first tile row this core signals the writers of the q / k columns it
    // read (cs_targets), and before its conv_state_out copies it waits for its own readers (cs_reads).
    uint32_t cs_signal_unit = 0xffffffffu;
    uint32_t cs_targets = 0;
    uint32_t cs_target_x[GDN_MAX_CS_SIGNAL_TARGETS];
    uint32_t cs_target_y[GDN_MAX_CS_SIGNAL_TARGETS];
    uint32_t cs_reads = 0;
    if constexpr (CS_IN_PLACE) {
        cs_signal_unit = get_arg_val<uint32_t>(arg++);
        cs_targets = get_arg_val<uint32_t>(arg++);
        for (uint32_t t = 0; t < cs_targets; ++t) {
            cs_target_x[t] = get_arg_val<uint32_t>(arg++);
            cs_target_y[t] = get_arg_val<uint32_t>(arg++);
        }
        cs_reads = get_arg_val<uint32_t>(arg++);
    }
    const uint32_t reader_units = STATE_SPLIT ? get_arg_val<uint32_t>(arg++) : 0;
    const uint32_t reader_share = STATE_SPLIT ? get_arg_val<uint32_t>(arg++) : 0;
    // S0_WRITER: the writer reads this core's S0 (every unit's) on its own NoC, this RISC none.
    const bool s0_by_writer = GDN_S0_WRITER && get_arg_val<uint32_t>(arg++) != 0;

    // SMALL_MCAST: A_log, dt_bias, b, a and norm_weight land from the key head's q core (host SMALL_INPUTS_MCAST) and
    // are copied from there into the unit's slots, under INPUTS_TRID before the inputs flush.
    // The channel's named args exist only in SMALL_MCAST programs.
    constexpr bool SMALL_MCAST = GDN_SMALL_MCAST != 0;
#if GDN_SMALL_MCAST
    constexpr auto small_channel = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("qks_ct_offset"), get_named_compile_time_arg_val("qks_rt_offset")>();
#endif
    const uint32_t small_base = SMALL_MCAST ? get_write_ptr(GDN_CB_SMALL_IN) : 0;
    bool small_landed = false;
    uint32_t kq_rows_landed = 0;  // PREP_SHARE_QK prep core: tile rows of qn / kn handed to compute
    uint32_t prev_head = 0xffffffffu;
    uint32_t prev_norm_w = 0;
    uint32_t prev_conv_w = 0;
    uint32_t prev_gate_w = 0;
    uint32_t prev_gate_in = 0;
    for (uint32_t pass = 0; pass < 2; ++pass) {
        if (CS_IN_PLACE && pass == 1) {
            Semaphore<>(GDN_CS_READ_SEM_ID).wait_min(cs_reads);
        }
        for (uint32_t unit_idx = 0; unit_idx < num_units; ++unit_idx) {
            const uint32_t unit = first_unit + unit_idx * unit_stride;
            GDN_ZONE("gdn_rd_unit");
            const uint32_t pair = unit / PIECES;
            const uint32_t v_first = PREP || PREP_QK ? 0 : (unit % PIECES) * Vb;  // first v / state column of the piece
            const bool p_half = SPLIT && pair / Hv >= SEGS;
            const uint32_t seq = p_half ? pair / Hv - (SEGS - 1) : pair / Hv;
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
            // Decode: the first window group's staged current and history rows, read into staging. They are what the
            // window waits for, so they are issued before the weights and gates (transaction id WINDOW_TRID): the row
            // copies start as soon as they land, while the weights are still in flight.
            const uint32_t seq_first = seq * T;
            const uint32_t hist_row0 = GDN_HISTORY * seq;
            const bool unit_hist_staged = (hist_row0 % TILE_ROWS) + GDN_HISTORY <= TILE_ROWS;
            // Unit column `local` into staging slot `slot`: its current row `row` of tile row `tr`, and (with_hist) its
            // history rows.
            auto issue_staged = [&](uint32_t local, uint32_t slot, uint32_t tr, uint32_t row, bool with_hist) {
                const uint32_t ct = pair_col(unit_col(local));
                read_row_span(mixed, tr * Mt + ct, row, row + 1, cur_stage + slot * TILE_BYTES);
                if (with_hist) {
                    const uint32_t r0 = hist_row0 % TILE_ROWS;
                    read_row_span(
                        conv_state,
                        (hist_row0 / TILE_ROWS) * Ct + ct,
                        r0,
                        r0 + GDN_HISTORY,
                        hist_stage + slot * TILE_BYTES);
                }
            };
            const bool prefetch = DECODE && !STATE && pass == 0;
            // SMALL_MCAST: slots awaiting their landed small inputs (0: none).
            uint32_t small_norm_w_dst = 0, small_gate_w_dst = 0, small_gate_in_dst = 0;
            const bool early_s0 =
                EARLY_S0 && (GDN_EARLY_S0 == 1 || unit_idx > 0) && !PREP && prefetch && !s0_by_writer;
            if (prefetch) {
                noc_async_read_set_trid(WINDOW_TRID);
                for (uint32_t col = 0; col < group_cols; ++col) {
                    issue_staged(col, col, seq_first / TILE_ROWS, seq_first % TILE_ROWS, unit_hist_staged);
                }
                // EARLY_S0: S0 issued here lands while the weights and gate rows queue, but competes with them.
                if (early_s0) {
                    noc_async_read_set_trid(S0_TRID);
                    cb_reserve_back(cb_state_in, n_state);
                    cb_reserve_back(cb_state_in_mm, n_state);
                    read_s0(get_write_ptr(cb_state_in), pair, v_first);
                }
                noc_async_read_set_trid(INPUTS_TRID);  // until the first window group's flush
            }
            if (pass == 0) {
                // ---- load_pair_block ----
                // norm_weight columns of the unit (a prep core's r heads share the Vt tiles).
                constexpr uint32_t norm_w_pages = PREP ? Vt : Vb;
                const bool reuse = !PREP && head * PIECES + v_first / Vb == prev_head;
                prev_head = head * PIECES + v_first / Vb;
                if constexpr (!STATE && !PREP_QK) {
                    cb_reserve_back(cb_norm_w, norm_w_pages);
                    const uint32_t norm_w_base = get_write_ptr(cb_norm_w);
                    if (SMALL_MCAST && !reuse) {
                        small_norm_w_dst = norm_w_base;
                    }
                    for (uint32_t vt = 0; vt < norm_w_pages && !reuse && !SMALL_MCAST; ++vt) {
                        read_row_span(norm_w, v_first + vt, 0, 1, norm_w_base + vt * TILE_BYTES);
                    }
                    if (reuse) {
                        noc_async_read(get_noc_addr(prev_norm_w), norm_w_base, norm_w_pages * TILE_BYTES);
                    }
                    prev_norm_w = norm_w_base;
                }
                if constexpr (!STATE) {
                    cb_reserve_back(cb_conv_w, conv_w_pages);
                    if constexpr (!PREP_QK) {
                        cb_reserve_back(cb_gate_w, 2);
                    }
                    const uint32_t conv_w_base = get_write_ptr(cb_conv_w);
                    const uint32_t gate_w_base = PREP_QK ? 0 : get_write_ptr(cb_gate_w);
                    if (reuse) {
                        noc_async_read(get_noc_addr(prev_conv_w), conv_w_base, conv_w_pages * TILE_BYTES);
                        noc_async_read(get_noc_addr(prev_gate_w), gate_w_base, 2 * TILE_BYTES);
                    }
                    prev_conv_w = conv_w_base;
                    prev_gate_w = gate_w_base;
                    for (uint32_t local = 0; local < unit_cols && !reuse; ++local) {
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
                    if (SMALL_MCAST && !reuse) {
                        small_gate_w_dst = gate_w_base;
                    } else if (!reuse && !PREP_QK) {
                        read_row_span(a_log, head / TILE_ROWS, 0, 1, gate_w_base);
                        read_row_span(dt_bias, head / TILE_ROWS, 0, 1, gate_w_base + TILE_BYTES);
                    }
                }
                // Completed by the first window group's flush (one DRAM round trip for weights, gates and window).
                bool weights_pending = !STATE;

                const uint32_t seq_lo = seq * T;
                const uint32_t seq_hi = seq_lo + T;
                const uint32_t first_tr = seq_lo / TILE_ROWS;
                const uint32_t last_tr = (seq_hi - 1) / TILE_ROWS;
                // SEQ_SPLIT: a segment's history rows are the previous segment's last token rows.
                const uint32_t hist_lo = SPLIT && seq > 0 ? 0 : seq_lo;
                // ---- z columns [first, first + count) of the unit, one tile row ----
                auto read_z = [&](uint32_t tr, uint32_t first, uint32_t count) {
                    const uint32_t base_row = tr * TILE_ROWS;
                    const uint32_t lo = seq_lo > base_row ? seq_lo : base_row;
                    const uint32_t hi = seq_hi < base_row + TILE_ROWS ? seq_hi : base_row + TILE_ROWS;
                    cb_reserve_back(cb_z, count);
                    const uint32_t z_base = get_write_ptr(cb_z);
                    for (uint32_t vt = 0; vt < count; ++vt) {
                        const uint32_t page = tr * z_row_tiles + z_col0 + head * Vt + v_first + first + vt;
                        read_tile_rows(rows, z_acc, page, base_row, lo, hi, z_base + vt * TILE_BYTES);
                    }
                    rows.finish();
                    cb_push_back(cb_z, count);
                    GDN_MARK("m_rd_z");
                };
                const uint32_t norm_units = (last_tr - first_tr + 1 - TAIL) * PREP_HEADS;
                uint32_t z_unit = norm_offset;  // NORM_SPLIT: next norm unit whose z is read
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
                            while (granted_rows < n_rows && granted_rows - i < GDN_GRANT_DEPTH &&
                                   cb_pages_reservable_at_back(cb_in_v, (granted_rows - i + 1) * Vb)) {
                                granted.up(
                                    noc, producer_x[granted_rows % producers], producer_y[granted_rows % producers], 1);
                                ++granted_rows;
                            }
                        };
                        if (granted_rows == i) {
                            cb_reserve_back(cb_in_v, Vb);  // row i's slot: freed once compute pops row i - INBOX_SLOTS
                        }
                        if (tr == first_tr) {
                            // S0 while the first rows are prepped: compute's first step needs it with row 0.
                            grant_free_slots();
                            cb_reserve_back(cb_state_in, n_state);
                            cb_reserve_back(cb_state_in_mm, n_state);
                            if (SPLIT && (p_half || seq > 0)) {
                                // [P | H] starts from [I | 0]: the P piece holds columns [v_first, v_first + Vb) of I.
                                CircularBuffer s0_buf(cb_state_in);
                                noc.async_write_zeros(s0_buf, n_state * STATE_TILE_BYTES, {.offset_bytes = 0});
                                noc.write_zeros_l1_barrier();
                                volatile tt_l1_ptr uint32_t* s0 =
                                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_state_in));
                                for (uint32_t c = 0; p_half && c < Vb; ++c) {
                                    const uint32_t tile = (v_first + c) * Vb + c;
                                    for (uint32_t r = 0; r < TILE_ROWS; ++r) {
                                        s0[tile * (STATE_TILE_BYTES / 4) + (r >> 4) * 768 + (r & 15) * 17] =
                                            0x3f800000u;
                                    }
                                }
                            } else {
                                read_s0(get_write_ptr(cb_state_in), pair, v_first);
                            }
                            noc_async_read_barrier();
                            cb_push_back(cb_state_in, n_state);
                            cb_push_back(cb_state_in_mm, n_state);
                        }
                        {
                            GDN_WAIT_ZONE("w_land");
                            do {
                                grant_free_slots();
                            } while (*landed < i / INBOX_SLOTS + 1);
                        }
                        GDN_MARK("m_rd_landed");
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
                        if (kq_landing) {
                            // ---- PREP_SHARE_QK: grant this tile row's cb_kq_rows slot to the q and k cores (free once
                            // compute has popped the row PREP_KQ_DEPTH rows back); the row is handed to compute after
                            // the other inputs, once both qn and kn have landed.
                            cb_reserve_back(cb_kq_rows, 2 * Kt);
                            Semaphore<> grant(kq_grant_sem);
                            grant.up(noc, kq_src[0], kq_src[1], 1);
                            grant.up(noc, kq_src[2], kq_src[3], 1);
                        }
                        // ---- gate inputs: token rows of the b, a tiles of the gate column ----
                        // Packed form: the gate tile is read as the a tile; b is split from it once it has landed.
                        uint32_t packed_gate_lo = 0, packed_gate_hi = 0;
                        if constexpr (!PREP_QK) {
                            cb_reserve_back(cb_gate_in, 2);
                            const uint32_t gate_in_base = get_write_ptr(cb_gate_in);
                            const uint32_t gate_page = tr * gate_row_tiles + gate_col0 + head / TILE_ROWS;
                            // Decode, units of consecutive sequences: the first unit reads the gate rows of all of
                            // them.
                            const bool shared_gates = T == 1 && unit_stride == Hv * PIECES;
                            if (reuse && shared_gates) {
                                noc_async_read(get_noc_addr(prev_gate_in), gate_in_base, 2 * TILE_BYTES);
                            } else if (SMALL_MCAST) {
                                small_gate_in_dst = gate_in_base;
                            } else {
                                uint32_t gate_hi = hi - base_row;
                                if (shared_gates) {
                                    gate_hi = lo - base_row + num_units - unit_idx;
                                    gate_hi = gate_hi < TILE_ROWS ? gate_hi : TILE_ROWS;
                                }
                                if constexpr (PACKED) {
                                    packed_gate_lo = lo - base_row;
                                    packed_gate_hi = gate_hi;
                                } else {
                                    read_row_span(b_acc, gate_page, lo - base_row, gate_hi, gate_in_base);
                                }
                                read_row_span(a_acc, gate_page, lo - base_row, gate_hi, gate_in_base + TILE_BYTES);
                            }
                            prev_gate_in = gate_in_base;
                        }
                        bool gates_pending = !PREP_QK;  // pushed after the first window group's flush

                        // ---- conv window: X_j[p] = xx[row(p) + j - 3], WINDOW_GROUP_COLS column tiles per hand-off
                        // ---- Per group: one zero fill, one flush of every DRAM row read (current rows + history
                        // rows), then one flush of the L1 row shifts that source the current-row tiles. A uniform group
                        // size keeps every reservation contiguous in cb_window (its size is a multiple of a group).
                        const uint32_t group_tiles = GDN_CONV_TAPS * group_cols;
                        for (uint32_t group_first = 0; group_first < unit_cols; group_first += group_cols) {
                            {
                                GDN_WAIT_ZONE("w_slot");
                                cb_reserve_back(cb_window, group_tiles);
                            }
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
                            const bool hist_needed = lo < hist_lo + GDN_HISTORY;
                            const bool hist_staged = hist_needed && unit_hist_staged;
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
                                const bool prefetched = prefetch && group_first == 0;
                                if (cur_staged) {
                                    if (!prefetched) {
                                        issue_staged(group_first + col, col, tr, lo - base_row, hist_staged);
                                    }
                                } else {
                                    read_tile_rows(
                                        rows,
                                        mixed,
                                        tr * Mt + ct,
                                        base_row,
                                        lo,
                                        hi,
                                        col_win + (GDN_CONV_TAPS - 1) * TILE_BYTES);
                                }
                                if (hist_staged && !cur_staged) {
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
                                        (prev_row0 / TILE_ROWS) * Mt + ct,
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
                                        if (row >= hist_lo + shift) {
                                            const uint32_t src = row - shift;
                                            rows.read(mixed, (src / TILE_ROWS) * Mt + ct, src % TILE_ROWS, tap_tile, p);
                                        } else if (!hist_staged) {
                                            const uint32_t src = hist_row0 + (row - seq_lo) + tap;
                                            rows.read(
                                                conv_state, (src / TILE_ROWS) * Ct + ct, src % TILE_ROWS, tap_tile, p);
                                        }
                                    }
                                }
                            }
                            const bool split_flush = prefetch && group_first == 0;
                            if (split_flush) {
                                noc_async_read_barrier_with_trid(WINDOW_TRID);
                                GDN_MARK("m_rd_win");
                                noc_async_read_set_trid(
                                    WINDOW_TRID);  // the row copies, so the inputs barrier skips them
                            } else {
                                rows.finish();
                            }
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
                            if (split_flush) {
                                if (SMALL_MCAST && (small_norm_w_dst | small_gate_w_dst | small_gate_in_dst) != 0) {
#if GDN_SMALL_MCAST
                                    if (!small_landed) {
                                        small_channel.optional_receiver(noc)->receive();
                                        small_landed = true;
                                        GDN_MARK("m_rd_small");
                                    }
#endif
                                    noc_async_read_set_trid(INPUTS_TRID);
                                    if (small_gate_w_dst != 0) {  // A_log, dt_bias
                                        noc_async_read(get_noc_addr(small_base), small_gate_w_dst, 2 * TILE_BYTES);
                                    }
                                    if (small_gate_in_dst != 0) {  // b, a
                                        noc_async_read(
                                            get_noc_addr(small_base + 2 * TILE_BYTES),
                                            small_gate_in_dst,
                                            2 * TILE_BYTES);
                                    }
                                    if (small_norm_w_dst != 0) {
                                        noc_async_read(
                                            get_noc_addr(small_base + (GDN_SMALL_IN_GATE_PAGES + v_first) * TILE_BYTES),
                                            small_norm_w_dst,
                                            Vb * TILE_BYTES);
                                    }
                                }
                                noc_async_read_barrier_with_trid(INPUTS_TRID);  // weights and gate rows
                                GDN_MARK("m_rd_inputs");
                                noc_async_read_set_trid(0);
                                if (GDN_S0_WRITER_AFTER_INPUTS && s0_by_writer && unit_idx == 0) {
                                    cb_reserve_back(GDN_CB_S0_GO, 1);
                                    cb_push_back(GDN_CB_S0_GO, 1);
                                }
                                rows.finish_staged();
                            }
                            if (weights_pending) {
                                cb_push_back(cb_conv_w, conv_w_pages);
                                if constexpr (!PREP_QK) {
                                    cb_push_back(cb_gate_w, 2);
                                    cb_push_back(cb_norm_w, norm_w_pages);
                                }
                                weights_pending = false;
                            }
                            if (gates_pending) {
                                if (packed_gate_hi != 0) {
                                    const uint32_t gate_in = get_write_ptr(cb_gate_in);
                                    gdn::packed_gate_b<TILE_BYTES>(
                                        gate_in + TILE_BYTES, gate_in, Hv, packed_gate_lo, packed_gate_hi);
                                }
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
                            GDN_MARK("m_rd_window_pushed");
                        }
                        // History rows are read in the sequence's first tile row only.
                        if (CS_IN_PLACE && tr == first_tr && unit_idx == cs_signal_unit) {
                            noc_async_read_barrier();
                            Semaphore<> cs_read(GDN_CS_READ_SEM_ID);
                            for (uint32_t t = 0; t < cs_targets; ++t) {
                                cs_read.up(noc, cs_target_x[t], cs_target_y[t], 1);
                            }
                        }
                    }
                    // ---- S0 of the piece (Kt x Vb slice of the pair state), after the first tile row's prep inputs
                    // ----
                    if (tr == first_tr && !PREP && !PREP_QK && !STATE && !s0_by_writer) {
                        if (early_s0) {
                            noc_async_read_barrier_with_trid(S0_TRID);
                        } else {
                            cb_reserve_back(cb_state_in, n_state);
                            cb_reserve_back(cb_state_in_mm, n_state);
                            read_s0(get_write_ptr(cb_state_in), pair, v_first);
                            noc_async_read_barrier();
                        }
                        cb_push_back(cb_state_in, n_state);
                        cb_push_back(cb_state_in_mm, n_state);
                        GDN_MARK("m_rd_s0");
                    }

                    if constexpr (STATE && TAIL != 0) {
                        if (tr == last_tr) {
                            // The piece's norm_weight columns, then z of the rows the state core normalizes.
                            cb_reserve_back(cb_norm_w, Vb);
                            for (uint32_t vt = 0; vt < Vb; ++vt) {
                                read_row_span(norm_w, v_first + vt, 0, 1, get_write_ptr(cb_norm_w) + vt * TILE_BYTES);
                            }
                            noc_async_read_barrier();
                            cb_push_back(cb_norm_w, Vb);
                            for (uint32_t t2 = last_tr + 1 - TAIL; t2 <= last_tr; ++t2) {
                                read_z(t2, 0, Vb);
                            }
                        }
                    }
                    if constexpr (STATE && SPLIT) {
                        if (tr == last_tr && !p_half && seq > 0) {
                            // P and the entering state, then the tile rows of Oh_P (SEG_ROW_DEPTH in flight).
                            const uint32_t n_rows = last_tr - first_tr + 1;
                            Semaphore<>(SEG_SEM_ID).wait_min(PIECES + 1);
                            cb_reserve_back(cb_seg_in, Kt * Kt + n_state);
                            cb_push_back(cb_seg_in, Kt * Kt + n_state);
                            uint32_t credits = 0;
                            for (uint32_t i = 0; i < n_rows; ++i) {
                                if (credits == i) {
                                    cb_reserve_back(cb_seg_row, Kt);
                                }
                                while (credits < n_rows &&
                                       cb_pages_reservable_at_back(cb_seg_row, (credits - i + 1) * Kt)) {
                                    for (uint32_t p = 0; p < PIECES; ++p) {
                                        Semaphore<>(SEG_SEM_ID + v_first / Vb).up(noc, seg_peer_x[p], seg_peer_y[p], 1);
                                    }
                                    ++credits;
                                }
                                for (uint32_t p = 0; p < PIECES; ++p) {
                                    Semaphore<>(SEG_SEM_ID + 1 + p).wait_min(i + 1);
                                }
                                cb_reserve_back(cb_seg_row, Kt);
                                cb_push_back(cb_seg_row, Kt);
                            }
                        }
                    }
                    if constexpr (STATE || PREP_QK) {
                        continue;
                    }
                    if constexpr (!NORM_SPLIT) {
                        read_z(tr, 0, Vb);
                    }
                    if (kq_landing) {
                        ++kq_rows_landed;
                        Semaphore<>(QN_LANDED_SEM_ID).wait_min(kq_rows_landed);
                        Semaphore<>(KN_LANDED_SEM_ID).wait_min(kq_rows_landed);
                        cb_push_back(cb_kq_rows, 2 * Kt);
                    }
                    if constexpr (NORM_SPLIT) {
                        // norm_mid: z of the units compute normalizes before this row's heads, in compute's order.
                        for (uint32_t m = 0; m < norm_mid && tr != first_tr + tr_offset && z_unit < norm_units; ++m) {
                            read_z(first_tr + z_unit / PREP_HEADS, (z_unit % PREP_HEADS) * Vt, Vt);
                            z_unit += norm_stride;
                        }
                    }
                }
                if constexpr (NORM_SPLIT) {
                    for (; z_unit < norm_units; z_unit += norm_stride) {
                        read_z(first_tr + z_unit / PREP_HEADS, (z_unit % PREP_HEADS) * Vt, Vt);
                    }
                }

                continue;
            }
            if constexpr (PREP || PREP_QK || CS_BY_QK_CORES) {
                continue;
            }
            // ---- copy_conv_state_block: conv_state_out[3b + n] = xx[b, T + n] ----
            // Columns are staged HISTORY rows apiece in the rows of one scratch tile: one read flush and one
            // write barrier per CS_COLS_PER_FLUSH columns.
            if (SPLIT && (p_half || seq + 1 != SEGS)) {
                continue;  // SEQ_SPLIT: the last segment's rows are the sequence's last
            }
            const uint32_t cs_seq = SPLIT ? 0 : seq;
            const uint32_t scratch = get_write_ptr(cb_scratch);
            const uint32_t seq_lo = seq * T;
            // q/k columns are written by one unit per key head (first head, first piece).
            const uint32_t first_idx = (CS_SPLIT == 0 && head % ratio == 0 && v_first == 0) ? 0 : 2 * Kt;
            for (uint32_t flush_first = first_idx; flush_first < pair_cols; flush_first += CS_COLS_PER_FLUSH) {
                const uint32_t flush_end =
                    pair_cols < flush_first + CS_COLS_PER_FLUSH ? pair_cols : flush_first + CS_COLS_PER_FLUSH;
                for (uint32_t idx = flush_first; idx < flush_end; ++idx) {
                    const uint32_t ct = pair_col(idx);
                    const uint32_t slot_row = (idx - flush_first) * GDN_HISTORY;
                    for (uint32_t n = 0; n < GDN_HISTORY; ++n) {
                        if (T + n < GDN_HISTORY) {
                            const uint32_t src = GDN_HISTORY * cs_seq + T + n;
                            rows.read(conv_state, (src / TILE_ROWS) * Ct + ct, src % TILE_ROWS, scratch, slot_row + n);
                        } else {
                            const uint32_t src = seq_lo + T + n - GDN_HISTORY;
                            rows.read(mixed, (src / TILE_ROWS) * Mt + ct, src % TILE_ROWS, scratch, slot_row + n);
                        }
                    }
                }
                rows.finish();
                for (uint32_t idx = flush_first; idx < flush_end; ++idx) {
                    const uint32_t ct = pair_col(idx);
                    const uint32_t slot_row = (idx - flush_first) * GDN_HISTORY;
                    for (uint32_t n = 0; n < GDN_HISTORY; ++n) {
                        const uint32_t dst = GDN_HISTORY * cs_seq + n;
                        write_row(conv_state_out, scratch, slot_row + n, (dst / TILE_ROWS) * Ct + ct, dst % TILE_ROWS);
                    }
                }
                noc_async_write_barrier();
            }
        }
    }

    // Only after every read: the host gives a core a share only if cb_state_out holds all of its states, so compute
    // never waits on these writes for space.
    if (STATE_SPLIT && !PREP && !PREP_QK && !STATE && (reader_units != 0 || reader_share != 0)) {
        // The k-th state (unit k) occupies cb_state_out tiles [k n_state, (k + 1) n_state) modulo its size.
        const auto& state_cb = get_local_cb_interface(cb_state_out);
        const uint32_t fifo_base = state_cb.fifo_limit - state_cb.fifo_size;
        const uint32_t fifo_tiles = state_cb.fifo_size / STATE_TILE_BYTES;
        const uintptr_t pushed_ptr = (uintptr_t)get_cb_tiles_received_ptr(cb_state_out);
        for (uint32_t unit_idx = 0; unit_idx < num_units; ++unit_idx) {
            const uint32_t unit = first_unit + unit_idx * unit_stride;
            const uint32_t first_page = (unit / PIECES) * Kt * Vt + (unit % PIECES) * Vb;
            for (uint32_t j = 0; j < n_state; ++j) {
                if (!gdn::state_tile_on_reader(unit_idx, j, reader_units, reader_share)) {
                    continue;
                }
                const uint32_t tile = unit_idx * n_state + j;
                while ((uint16_t)((uint16_t)reg_read(pushed_ptr) - state_out_pushed0) <= tile) {
                }
                const uint32_t page = gdn::state_out_page<Kt, Vt, Vb>(first_page, j, GDN_STATE_STAGGER);
                noc_async_write(
                    fifo_base + (tile % fifo_tiles) * STATE_TILE_BYTES, state_out.get_noc_addr(page), STATE_TILE_BYTES);
            }
            noc_async_write_barrier();
            cb_reserve_back(cb_state_done, 1);
            cb_push_back(cb_state_done, 1);
        }
    }
}
