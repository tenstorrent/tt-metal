// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gen_qkv_causal_conv1d_silu reader (NCRISC / NoC0) — perf experiment `dynamic_split_reads`.
//
// Copy of the op's reader with ONE change: the ROW_MAJOR split reader no longer hands the writer
// RISC a static share (body rows [32 - split_rows, 32)) of every tile-row. Instead the 32 body
// sticks of each tile-row are CLAIMED dynamically: this kernel from the front, the writer RISC
// (BRISC / NoC1) from the back, each gated by its own NoC's completion rate (at most READ_WINDOW
// of this RISC's reads outstanding), so whichever NoC is faster on this core takes more sticks, and
// the writer only helps while it is idle (it polls only while waiting for compute's output).
//
// Shared L1 words (single writer each, monotone within a tile-row; the reading RISC fences first):
//   R = *(cb_split_requests page + 0)  this kernel:  seq << 16 | front << 8 | {OPEN, ACK, CLOSED}
//   D = *(cb_split_requests page + 16) this kernel:  the tile-row's L1 destination (written before R)
//   W = *(cb_split_done page + 0)      writer:       seq << 16 | back << 8  | {ENTER, CLAIM, DONE}
//   N = *(cb_split_done page + 4)      writer:       its NoC1 read-issue count at DONE (written before W)
// Per tile-row seq (1, 2, ... per core): R = (seq, 0, OPEN). The writer may only read sticks after
// this kernel ACKs its ENTER. This kernel is the arbiter: it ACKs an ENTER while sticks remain
// (and then waits for the writer's DONE before pushing), or, having claimed all 32 sticks without
// acking, stores CLOSED and pushes at once (the writer, spinning on R, abandons the row). An acked
// writer stops claiming when its own output becomes ready and stores DONE without a barrier; this
// kernel then claims the rest and waits until the core's NoC1 read-response counter reaches N. So
// no stick is read after its slot is pushed, and no atomics are needed. Both sides may claim the
// meeting chunk (stale view of the other's counter): a duplicate read of the same stick into the
// same L1 address writes identical bytes, and both land before the push. The core's first tile-row
// is opened pre-ACKed (the writer has no output to wait for before it, so it is guaranteed to
// help; and this kernel blocks on the tap multicast inside that row).
//
// Block operations owned here (op_design.md -> Block schedule): load_taps (multicast per group),
// load_rows_block (halo + body sticks), as in the op's reader.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

// Stage zones (MaybeDeviceZoneScope, PERMANENT — perf tournaments rank stages from them; free when the
// profiler is off): reader_taps_issue (group sender, once), per block reader_rows_reserve (back-pressure
// from compute), reader_halo_issue; per tile-row reader_rows_issue, reader_rows_barrier (incl. the tap
// publish on the first row), reader_split_wait (the writer's NoC1 share); TILE regime reader_tile_issue,
// reader_untilized_wait, reader_slot_copy.
// Measurement-only ablation (compiled out unless defined; wrong results by construction):
// GQKV_ABLATE_READS drops the input / history / halo DRAM reads (taps, CB sync and barriers stay).
#ifdef GQKV_ABLATE_READS
#define GQKV_READ(...)
#else
#define GQKV_READ(...) noc_async_read(__VA_ARGS__)
#endif

namespace {

constexpr uint32_t cb_rows = 0;
constexpr uint32_t cb_tap_tiles = 1;
constexpr uint32_t cb_input_tiles = 4;
constexpr uint32_t cb_untilized_rows = 5;
constexpr uint32_t cb_split_requests = 6;  // split reader: this kernel -> writer, one L1 row address per tile-row
constexpr uint32_t cb_split_done = 7;      // split reader: writer -> this kernel, one token per tile-row

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;
constexpr uint32_t TAP_COUNT = 4;
constexpr uint32_t HALO_ROWS = TAP_COUNT - 1;
constexpr uint32_t TAP_ROW_STRIDE = 2;  // tap p lives in row 2p of the packed tap tile (64 B aligned)

// Dynamic split states (low byte of R / W).
constexpr uint32_t R_OPEN = 0;
constexpr uint32_t R_ACK = 1;
constexpr uint32_t R_CLOSED = 2;
constexpr uint32_t W_ENTER = 1;
constexpr uint32_t W_CLAIM = 2;
constexpr uint32_t W_DONE = 3;

// This RISC's reads still in flight on its NoC (issued - responses received; wraps consistently).
FORCE_INLINE uint32_t reads_outstanding() {
    return noc_reads_num_issued[noc_index] - NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED);
}

// Self-aimed L1 -> L1 copy (dataflow_kernel_lib::local_addr: the source is this core's own L1).
FORCE_INLINE void copy_local(const Noc& noc, uint32_t src_addr, uint32_t dst_addr, uint32_t size_bytes) {
    UnicastEndpoint self_ep;
    noc.async_read(
        self_ep,
        CoreLocalMem<uint32_t>(dst_addr),
        size_bytes,
        dataflow_kernel_lib::local_addr(src_addr, noc.get_noc_id()),
        {});
}

// Row 0 of tap tile (0, ch_group_tile_start + c) -> row 2*tap_idx of packed tile c.
template <uint32_t block_ch_tiles, uint32_t tile_bytes, uint32_t elem_bytes, typename Accessor>
FORCE_INLINE void issue_tap_rows(
    const Accessor& tap_acc, uint32_t tap_idx, uint32_t tap_base, uint32_t ch_group_tile_start) {
    constexpr uint32_t face_row_bytes = FACE_DIM * elem_bytes;
    constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * elem_bytes;
    const uint32_t row_offset = TAP_ROW_STRIDE * tap_idx * face_row_bytes;
    for (uint32_t c = 0; c < block_ch_tiles; ++c) {
        const uint64_t src = tap_acc.get_noc_addr(ch_group_tile_start + c);
        const uint32_t dst = tap_base + c * tile_bytes + row_offset;
        noc_async_read(src, dst, face_row_bytes);                            // face 0: channels 0..15
        noc_async_read(src + face_bytes, dst + face_bytes, face_row_bytes);  // face 1: channels 16..31
    }
}

// History halo: the 3 history rows of channels [c0, c0 + 32 * block_ch_tiles) -> sticks 0..2 of the slot.
//   token-paged   (history_page_channels == 0): [1, 3, C]      row r = page r, channel c at element c.
//   channel-paged (history_page_channels == P): [1, C/P, 3P]   row r of channel c = page c / P,
//                 element r * P + c % P. A group whose channel slice straddles pages issues one read
//                 per contiguous run per row; every run starts on a multiple of 32 channels (c0 and P
//                 are tile multiples), i.e. 64 B aligned in source and destination.
template <uint32_t block_ch_tiles, uint32_t history_page_channels, uint32_t elem_bytes, typename Accessor>
FORCE_INLINE void read_history_halo(const Accessor& history_acc, uint32_t slot_base, uint32_t ch_group_tile_start) {
    constexpr uint32_t group_channels = block_ch_tiles * TILE_DIM;
    constexpr uint32_t stick_bytes = group_channels * elem_bytes;
    const uint32_t c0 = ch_group_tile_start * TILE_DIM;
    for (uint32_t r = 0; r < HALO_ROWS; ++r) {
        const uint32_t dst_row = slot_base + r * stick_bytes;
        if constexpr (history_page_channels == 0) {
            GQKV_READ(history_acc.get_noc_addr(r, c0 * elem_bytes), dst_row, stick_bytes);
        } else {
            constexpr uint32_t P = history_page_channels;
            for (uint32_t c = c0; c < c0 + group_channels;) {
                const uint32_t page = c / P;
                const uint32_t in_page = c - page * P;
                const uint32_t run_end = (page + 1) * P < c0 + group_channels ? (page + 1) * P : c0 + group_channels;
                GQKV_READ(
                    history_acc.get_noc_addr(page, (r * P + in_page) * elem_bytes),
                    dst_row + (c - c0) * elem_bytes,
                    (run_end - c) * elem_bytes);
                c = run_end;
            }
        }
    }
}

}  // namespace

void kernel_main() {
    // ---- compile-time args ----
    constexpr uint32_t block_ch_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_rows = get_compile_time_arg_val(1);
    constexpr bool tile_input = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t input_width_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t history_page_channels = get_compile_time_arg_val(6);  // 0: token-paged history
    constexpr uint32_t split_rows = get_compile_time_arg_val(7);             // > 0: the dynamic split is on
    [[maybe_unused]] constexpr bool stagger_row_walk = get_compile_time_arg_val(8) == 1;  // unused here
    constexpr uint32_t read_window = get_compile_time_arg_val(9);   // max NoC0 reads outstanding when claiming
    constexpr uint32_t claim_chunk = get_compile_time_arg_val(10);  // sticks per claim (divides 32)
    constexpr uint32_t head_sticks = get_compile_time_arg_val(11);  // sticks [0, head) claimed under read_window
    constexpr uint32_t tail_window = get_compile_time_arg_val(12);  // window for sticks beyond the head
    constexpr uint32_t dup_grace =
        get_compile_time_arg_val(13);  // polls to wait for the writer before re-reading its share
    constexpr uint32_t rows_depth = get_compile_time_arg_val(14);  // cb_rows slots (slot-reuse guard ring)
    constexpr auto input_args = TensorAccessorArgs<15>();
    constexpr auto history_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto tap0_args = TensorAccessorArgs<history_args.next_compile_time_args_offset()>();
    constexpr auto tap1_args = TensorAccessorArgs<tap0_args.next_compile_time_args_offset()>();
    constexpr auto tap2_args = TensorAccessorArgs<tap1_args.next_compile_time_args_offset()>();
    constexpr auto tap3_args = TensorAccessorArgs<tap2_args.next_compile_time_args_offset()>();
    constexpr auto taps_mcast = dataflow_kernel_lib::McastArgs<
        get_named_compile_time_arg_val("taps_mcast_ct_offset"),
        get_named_compile_time_arg_val("taps_mcast_rt_offset")>();

    constexpr uint32_t channel_tile_bytes = TILE_DIM * elem_bytes;  // one channel tile of one token row
    constexpr uint32_t stick_bytes = block_ch_tiles * channel_tile_bytes;
    constexpr uint32_t untilized_row_bytes = TILE_DIM * stick_bytes;  // one untilized tile-row = 32 sticks
    constexpr uint32_t first_row_push = TILE_DIM + HALO_ROWS;         // halo + first tile-row
    constexpr uint32_t tap_tiles_bytes = block_ch_tiles * tile_bytes;
    constexpr bool dyn_split = split_rows > 0;
    static_assert(split_rows <= TILE_DIM, "split_rows is a row count inside one tile-row");
    static_assert(split_rows == 0 || !tile_input, "the split reader is a ROW_MAJOR-regime lever");
    static_assert(claim_chunk > 0 && TILE_DIM % claim_chunk == 0, "claims tile the 32 body sticks");
    static_assert(read_window >= claim_chunk && tail_window >= claim_chunk, "a claim must fit the read window");
    static_assert(head_sticks % claim_chunk == 0, "the head is whole claims");
    static_assert(rows_depth >= 2 && rows_depth <= 4, "slot-reuse guard ring");
    constexpr bool speculative = dup_grace != 0xFFFFFFFF;  // may push before the writer's reads land

    // ---- runtime args ----
    uint32_t arg = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t history_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap3_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t ch_group_tile_start = get_arg_val<uint32_t>(arg++);
    const uint32_t input_col_offset_tiles = get_arg_val<uint32_t>(arg++);
    const uint32_t core_seq_start = get_arg_val<uint32_t>(arg++);
    const uint32_t core_seq_tiles = get_arg_val<uint32_t>(arg++);

    const uint32_t num_blocks_this_core = (core_seq_tiles + block_rows - 1) / block_rows;
    const uint32_t last_block_rows = core_seq_tiles - (num_blocks_this_core - 1) * block_rows;

    const auto input_acc = TensorAccessor(input_args, input_addr);
    const auto history_acc = TensorAccessor(history_args, history_addr);
    Noc noc;

    // ---- dynamic split: shared words (see the head comment) ----
    volatile tt_l1_ptr uint32_t* R = nullptr;
    volatile tt_l1_ptr uint32_t* D = nullptr;
    volatile tt_l1_ptr uint32_t* W = nullptr;
    volatile tt_l1_ptr uint32_t* N = nullptr;
    uint32_t seq = 0;
    [[maybe_unused]] uint32_t dup_sticks = 0;  // profiler evidence: sticks re-read speculatively
    if constexpr (dyn_split) {
        cb_reserve_back(cb_split_requests, 1);
        R = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_split_requests));
        D = R + 4;  // + 16 B
        *D = 0;
        *R = 0;
        cb_push_back(cb_split_requests, 1);  // "R is initialized" (never popped: the page stays shared)
        cb_wait_front(cb_split_done, 1);     // "W is initialized"
        W = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_split_done));
        N = W + 1;
    }
    constexpr uint32_t writer_noc = 1 - noc_index;
    // True once the writer has finished tile-row `row` (DONE for it, or entered a later one: it handles
    // rows in order and always ends an acked row with DONE) and every read it issued up to its latest
    // DONE has landed (this core's NoC1 read-response counter reached N).
    [[maybe_unused]] auto writer_landed = [&](uint32_t row) -> bool {
        invalidate_l1_cache();
        const uint32_t w = *W;
        const uint32_t wseq = w >> 16;
        if (wseq < row || (wseq == row && (w & 0xff) != W_DONE)) {
            return false;
        }
        invalidate_l1_cache();
        return static_cast<int32_t>(NOC_STATUS_READ_REG(writer_noc, NIU_MST_RD_RESP_RECEIVED) - *N) >= 0;
    };
    // Slot-reuse guard: guard[k % rows_depth] = the last tile-row of block k that was pushed before the
    // writer's reads into it were known to have landed (0: none). Those late reads write the same bytes
    // the slot already holds, so they are harmless until the slot is refilled — which waits here.
    [[maybe_unused]] uint32_t guard[4] = {0, 0, 0, 0};

    // Byte offset of this core's channel slice inside a token row of the input (column window:
    // the convolution channels start input_col_offset_tiles tiles into the row).
    const uint32_t input_byte_offset = (input_col_offset_tiles + ch_group_tile_start) * channel_tile_bytes;

    // ---- load_taps: the group's sender issues the DRAM reads; receivers wait for the multicast ----
    auto tap_sender = taps_mcast.optional_sender(noc);
    auto tap_receiver = taps_mcast.optional_receiver(noc);
    cb_reserve_back(cb_tap_tiles, block_ch_tiles);
    const uint32_t tap_base = get_write_ptr(cb_tap_tiles);  // same L1 address on every core of the group
    if (tap_sender) {
        MaybeDeviceZoneScope("reader_taps_issue");
        issue_tap_rows<block_ch_tiles, tile_bytes, elem_bytes>(
            TensorAccessor(tap0_args, tap0_addr), 0, tap_base, ch_group_tile_start);
        issue_tap_rows<block_ch_tiles, tile_bytes, elem_bytes>(
            TensorAccessor(tap1_args, tap1_addr), 1, tap_base, ch_group_tile_start);
        issue_tap_rows<block_ch_tiles, tile_bytes, elem_bytes>(
            TensorAccessor(tap2_args, tap2_addr), 2, tap_base, ch_group_tile_start);
        issue_tap_rows<block_ch_tiles, tile_bytes, elem_bytes>(
            TensorAccessor(tap3_args, tap3_addr), 3, tap_base, ch_group_tile_start);
    }
    // Called once, after the core's first tile-row reads are in flight: those reads keep landing while
    // the taps are distributed. Leaves every outstanding read complete.
    bool taps_pending = true;
    auto publish_taps_and_barrier = [&]() {
        if (tap_sender) {
            noc_async_read_barrier();  // own tap rows (and the first tile-row) landed
            tap_sender->send(tap_base, tap_base, tap_tiles_bytes);
        } else if (tap_receiver) {
            tap_receiver->receive();
        }
        noc_async_read_barrier();
        cb_push_back(cb_tap_tiles, block_ch_tiles);
        taps_pending = false;
    };

    uint32_t prev_slot_base = 0;
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t rows_this_block = (block_idx + 1 == num_blocks_this_core) ? last_block_rows : block_rows;
        const uint32_t block_seq_start = core_seq_start + block_idx * block_rows;  // first tile-row of the block
        const uint32_t slot_sticks = TILE_DIM * rows_this_block + HALO_ROWS;

        // ---- load_rows_block: one slot ----
        {
            MaybeDeviceZoneScope("reader_rows_reserve");
            cb_reserve_back(cb_rows, slot_sticks);
        }
        const uint32_t slot_base = get_write_ptr(cb_rows);
        if constexpr (dyn_split && speculative) {
            const uint32_t g = guard[block_idx % rows_depth];
            if (g != 0) {
                MaybeDeviceZoneScope("reader_slot_guard");
                while (!writer_landed(g)) {
                }
                guard[block_idx % rows_depth] = 0;
            }
        }

        // Halo sticks [0, 3): tokens block_t0 - 3 .. block_t0 - 1.
        bool halo_from_untilized = false;
        {
            MaybeDeviceZoneScope("reader_halo_issue");
            if (block_seq_start == 0) {
                read_history_halo<block_ch_tiles, history_page_channels, elem_bytes>(
                    history_acc, slot_base, ch_group_tile_start);
            } else if (block_idx > 0) {
                // The previous block is full-height: its last 3 body sticks sit at [32*block_rows, +3).
                copy_local(
                    noc, prev_slot_base + TILE_DIM * block_rows * stick_bytes, slot_base, HALO_ROWS * stick_bytes);
            } else if constexpr (!tile_input) {
                const uint32_t first_token = block_seq_start * TILE_DIM - HALO_ROWS;
                for (uint32_t r = 0; r < HALO_ROWS; ++r) {
                    GQKV_READ(
                        input_acc.get_noc_addr(first_token + r, input_byte_offset),
                        slot_base + r * stick_bytes,
                        stick_bytes);
                }
            } else {
                halo_from_untilized = true;  // rows 29..31 of the extra untilized tile-row t0/32 - 1
            }
        }

        if constexpr (!tile_input) {
            for (uint32_t m = 0; m < rows_this_block; ++m) {
                const uint32_t token0 = (block_seq_start + m) * TILE_DIM;
                const uint32_t dst = slot_base + (HALO_ROWS + TILE_DIM * m) * stick_bytes;
                if constexpr (!dyn_split) {
                    {
                        MaybeDeviceZoneScope("reader_rows_issue");
                        for (uint32_t i = 0; i < TILE_DIM; ++i) {
                            GQKV_READ(
                                input_acc.get_noc_addr(token0 + i, input_byte_offset),
                                dst + i * stick_bytes,
                                stick_bytes);
                        }
                    }
                    MaybeDeviceZoneScope("reader_rows_barrier");
                    if (taps_pending) {
                        publish_taps_and_barrier();
                    } else {
                        noc_async_read_barrier();
                    }
                } else {
                    // ---- dynamic claim of the 32 body sticks (front: this kernel; back: the writer) ----
                    ++seq;
                    const bool pre_acked = taps_pending;  // the core's first tile-row
                    bool acked = pre_acked;
                    *D = dst;
                    *R = (seq << 16) | (acked ? R_ACK : R_OPEN);
                    uint32_t f = 0;         // sticks [0, f) claimed here
                    uint32_t b = TILE_DIM;  // sticks [b, 32) claimed by the writer (last seen)
                    {
                        MaybeDeviceZoneScope("reader_rows_issue");
                        while (true) {
                            invalidate_l1_cache();
                            const uint32_t w = *W;
                            if ((w >> 16) == seq) {
                                if ((w & 0xff) == W_ENTER) {
                                    if (!acked && f < TILE_DIM) {
                                        acked = true;
                                        *R = (seq << 16) | (f << 8) | R_ACK;
                                    }
                                } else {
                                    b = (w >> 8) & 0xff;
                                }
                            }
                            if (f >= b) {
                                break;
                            }
                            // Head: claimed eagerly (bandwidth). Tail: only once this RISC's own reads have
                            // drained to tail_window, so an idle writer whose NoC is faster takes it first.
                            const uint32_t window = f < head_sticks ? read_window : tail_window;
                            if (reads_outstanding() + claim_chunk > window) {
                                if (taps_pending) {
                                    publish_taps_and_barrier();  // the first claims are in flight
                                }
                                continue;
                            }
                            for (uint32_t c = 0; c < claim_chunk; ++c) {
                                GQKV_READ(
                                    input_acc.get_noc_addr(token0 + f + c, input_byte_offset),
                                    dst + (f + c) * stick_bytes,
                                    stick_bytes);
                            }
                            f += claim_chunk;
                            *R = (seq << 16) | (f << 8) | (acked ? R_ACK : R_OPEN);
                        }
                    }
                    if (!acked) {
                        *R = (seq << 16) | (f << 8) | R_CLOSED;  // the writer never got in: it abandons the row
                    }
                    {
                        MaybeDeviceZoneScope("reader_rows_barrier");
                        if (taps_pending) {
                            publish_taps_and_barrier();
                        } else {
                            noc_async_read_barrier();
                        }
                    }
                    if (acked) {
                        // The writer's DONE carries no barrier: its share [b, 32) has landed once its NoC1
                        // responses reach N. Speculative mode: after dup_grace polls, re-read whatever share
                        // it still holds on this NoC and push when either copy has landed.
                        MaybeDeviceZoneScope("reader_split_wait");
                        uint32_t polls = 0;
                        while (!writer_landed(seq)) {
                            if constexpr (speculative) {
                                if (polls++ >= dup_grace) {
                                    break;
                                }
                            }
                        }
                        if constexpr (speculative) {
                            if (polls > dup_grace) {
                                // Stop the writer claiming (front = 32), then duplicate its share.
                                *R = (seq << 16) | (TILE_DIM << 8) | R_ACK;
                                invalidate_l1_cache();
                                const uint32_t w = *W;
                                const uint32_t b_dup = (w >> 16) == seq ? ((w >> 8) & 0xff) : b;
                                for (uint32_t i = b_dup; i < TILE_DIM; i += claim_chunk) {
                                    while (reads_outstanding() + claim_chunk > read_window) {
                                    }
                                    for (uint32_t c = 0; c < claim_chunk; ++c) {
                                        GQKV_READ(
                                            input_acc.get_noc_addr(token0 + i + c, input_byte_offset),
                                            dst + (i + c) * stick_bytes,
                                            stick_bytes);
                                    }
                                    dup_sticks += claim_chunk;
                                }
                                while (!ncrisc_noc_reads_flushed(noc_index) && !writer_landed(seq)) {
                                }
                                guard[block_idx % rows_depth] = seq;  // the slower copy may still be in flight
                            }
                        }
                    }
                }
                cb_push_back(cb_rows, m == 0 ? first_row_push : TILE_DIM);
            }
        } else {
            // Body: input tile-rows -> compute untilize -> cb_untilized_rows -> this slot.
            const uint32_t extra_rows = halo_from_untilized ? 1 : 0;
            const uint32_t rows_to_untilize = rows_this_block + extra_rows;
            const uint32_t first_tile_row = block_seq_start - extra_rows;
            auto issue_input_tile_row = [&](uint32_t tile_row) {
                MaybeDeviceZoneScope("reader_tile_issue");
                cb_reserve_back(cb_input_tiles, block_ch_tiles);
                const uint32_t dst = get_write_ptr(cb_input_tiles);
                const uint32_t first_page = tile_row * input_width_tiles + input_col_offset_tiles + ch_group_tile_start;
                for (uint32_t c = 0; c < block_ch_tiles; ++c) {
                    GQKV_READ(input_acc.get_noc_addr(first_page + c), dst + c * tile_bytes, tile_bytes);
                }
            };

            issue_input_tile_row(first_tile_row);
            if (taps_pending) {
                publish_taps_and_barrier();
            } else {
                noc_async_read_barrier();
            }
            cb_push_back(cb_input_tiles, block_ch_tiles);
            for (uint32_t j = 0; j < rows_to_untilize; ++j) {
                const bool next_pending = (j + 1 < rows_to_untilize);
                if (next_pending) {
                    issue_input_tile_row(first_tile_row + j + 1);  // in flight while we wait on row j
                }
                {
                    MaybeDeviceZoneScope("reader_untilized_wait");
                    cb_wait_front(cb_untilized_rows, block_ch_tiles);
                }
                MaybeDeviceZoneScope("reader_slot_copy");
                const uint32_t src = get_read_ptr(cb_untilized_rows);
                if (j < extra_rows) {
                    copy_local(noc, src + (TILE_DIM - HALO_ROWS) * stick_bytes, slot_base, HALO_ROWS * stick_bytes);
                } else {
                    const uint32_t m = j - extra_rows;
                    copy_local(noc, src, slot_base + (HALO_ROWS + TILE_DIM * m) * stick_bytes, untilized_row_bytes);
                }
                noc_async_read_barrier();
                if (next_pending) {
                    cb_push_back(cb_input_tiles, block_ch_tiles);
                }
                cb_pop_front(cb_untilized_rows, block_ch_tiles);
                if (j >= extra_rows) {
                    cb_push_back(cb_rows, j == extra_rows ? first_row_push : TILE_DIM);
                }
            }
        }
        prev_slot_base = slot_base;
    }
#if defined(PROFILE_KERNEL)
    if constexpr (dyn_split && speculative) {
        DeviceTimestampedData("reader_dup_sticks", dup_sticks);
    }
#endif
    // BB-03: nothing this kernel issued (reads, multicast writes, semaphore updates) is left in flight.
    noc_async_read_barrier();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
