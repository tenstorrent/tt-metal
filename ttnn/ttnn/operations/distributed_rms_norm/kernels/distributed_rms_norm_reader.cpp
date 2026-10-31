// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm reader (op_design.md -> Dataflow Strategy, Root reader as an event loop).
//
// Every compute core owns a W-slice [w0, w0 + V) of its tile-rows (R1: the whole row). Waves are numbered globally
// w = block * nwpb + k (k-th wave of the block, Rw rows; a short last block has ragged / empty waves).
//
// Prologue: cb_reduce_scaler <- 1.0 (or the [full, partial] pair for the sharded W-pad mask); sharded: cb_x_tiles IS
// the input shard (zero-copy), all of this core's rows are pushed once.
//
// Then ONE non-blocking poll loop; each cursor steps whenever its condition holds, so no wave waits behind an earlier
// wave's exchange (root order: distribute, collect, publish, gather — downstream first):
//   load        (interleaved) the next tile-row of x: V page reads issued, pushed in a later pass once they landed
//               (non-blocking), W-pad columns zeroed; READ_WAVE_GATE: wave k+1's rows only after wave k is reduced
//   member (R3, index > 0):
//     send      wave w's collapsed partials (cb_partial_col) -> the root's cb_gather slots [(k*G_w + index)*Rw, +n)
//               as faces 0/2, barrier, root SEM_GATHER_k += 1 (k = wave index within the block)
//     receive   SEM_INV counts the root's inv_rms writes into cb_inv_rms; push n per wave
//   root (index 0):
//     gather    (R3) own partial -> its gather slot 0, SEM_GATHER_k >= (G_w-1) per wave -> push G_w*Rw
//     publish   every wave of the call (empty ones too: the port counts every root every wave): column 0 of the
//               device partial (cb_dev_partial) -> 32 fp32 words per tile-row -> one NoC write into slot p of the
//               landing of each sending port of this root's link, barrier, port SEM_OWN_STATS += 1.
//               For wave w >= 1 only after the ports counted wave w-1 (SEM_OWN_COUNTED >= n_send * w).
//     collect   slot by slot (the ports notify per landed entry): own slot once published, each peer slot when its
//               entry is counted on SEM_LANDED_<d>; each record run is read from its port landing and scattered into
//               column 0 of tile t*G + q of cb_stats_gather; push G*n after the last slot (every chip sums identical
//               fp32 inputs in slot order)
//     distribute (R3) own copy into cb_inv_rms first; then per member row-run a linked multicast of faces 0/2 into
//               the members' cb_inv_rms + a multicast set of the cumulative wave count on SEM_INV (no ack barrier)
// Raw NoC / CB calls: no kernel_lib helper reads TILE pages with W-pad zeroing, moves a face-strided subset of a tile,
// extracts / scatters a tile's column 0, or polls several event sources without blocking (op_design.md -> Helpers
// considered and rejected).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "tools/profiler/kernel_profiler.hpp"

// Measurement-only per-stage zones (host DRMS_ZONES=1 -> define; never set in production).
#ifdef DRMS_ZONES
#define DRMS_ZONE(name) DeviceZoneScopedN(name)
#else
#define DRMS_ZONE(name)
#endif

namespace {

constexpr uint32_t cb_x_tiles = 1;
constexpr uint32_t cb_reduce_scaler = 2;
constexpr uint32_t cb_inv_rms = 4;
constexpr uint32_t cb_partial_col = 9;
constexpr uint32_t cb_gather = 10;
constexpr uint32_t cb_rms_root = 11;
constexpr uint32_t cb_dev_partial = 17;
constexpr uint32_t cb_stats_gather = 18;
constexpr uint32_t cb_stats_record = 19;

// Program semaphore ids (mirror the host). SEM_GATHER_BASE + k: member partial arrivals of wave k of a block — one
// counter per wave, because G_w - 1 members send each wave and a single cumulative counter could be satisfied by a fast
// member's wave k+1 before a slow member's wave k landed.
constexpr uint32_t SEM_INV = 0;
constexpr uint32_t SEM_OWN_STATS = 1;
constexpr uint32_t SEM_LANDED_FWD = 2;
constexpr uint32_t SEM_LANDED_BWD = 3;
constexpr uint32_t SEM_OWN_COUNTED = 4;
constexpr uint32_t SEM_GATHER_BASE = 5;
constexpr uint32_t MAX_WAVES_PER_BLOCK = 8;

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;
constexpr uint32_t RECORD_BYTES = TILE_DIM * 4;

FORCE_INLINE uint32_t sem_value(uint32_t id) {
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id));
}

// Element store of zero over n_elems elements (partial face rows only: <= 15 elements).
template <uint32_t es>
FORCE_INLINE void store_zeros(uint32_t addr, uint32_t n_elems) {
    if constexpr (es == 4) {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
        for (uint32_t i = 0; i < n_elems; ++i) {
            p[i] = 0;
        }
    } else {
        volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr);
        for (uint32_t i = 0; i < n_elems; ++i) {
            p[i] = 0;
        }
    }
}

// Zero columns [w_valid, 32) of the TILE at cb write pointer + tile_offset (base op mechanism): whole pad faces by the
// NoC zero loopback, partial face rows by element stores.
template <uint32_t es>
FORCE_INLINE void zero_tile_w_padding(
    const Noc& noc, const CircularBuffer& cb, uint32_t tile_offset, uint32_t w_valid) {
    constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * es;
    constexpr uint32_t face_row_bytes = FACE_DIM * es;
    const uint32_t tile_addr = cb.get_write_ptr() + tile_offset;
    if (w_valid <= FACE_DIM) {
        noc.async_write_zeros(cb, face_bytes, {.offset_bytes = tile_offset + 1 * face_bytes});
        noc.async_write_zeros(cb, face_bytes, {.offset_bytes = tile_offset + 3 * face_bytes});
        if (w_valid < FACE_DIM) {
            const uint32_t n = FACE_DIM - w_valid;
            for (uint32_t f = 0; f < 4; f += 2) {
                for (uint32_t r = 0; r < FACE_DIM; ++r) {
                    store_zeros<es>(tile_addr + f * face_bytes + r * face_row_bytes + w_valid * es, n);
                }
            }
        }
        noc.write_zeros_l1_barrier();
    } else {
        const uint32_t c0 = w_valid - FACE_DIM;
        const uint32_t n = FACE_DIM - c0;
        for (uint32_t f = 1; f < 4; f += 2) {
            for (uint32_t r = 0; r < FACE_DIM; ++r) {
                store_zeros<es>(tile_addr + f * face_bytes + r * face_row_bytes + c0 * es, n);
            }
        }
    }
}

// n interm tiles (column-0 valid) -> faces 0 and 2 only at (x, y, dst_l1): every consumer reads column 0 only (the
// lane-wise combine / finalize, the Col-broadcast scale), so faces 1/3 never move (base op rms_norm_combine.hpp).
template <uint32_t T>
FORCE_INLINE void write_faces_0_2(uint32_t src, uint32_t x, uint32_t y, uint32_t dst_l1, uint32_t n) {
    constexpr uint32_t F = T / 4;
    static_assert(F <= NOC_MAX_BURST_SIZE, "one face must fit one NoC packet");
    if (n == 0) {
        return;
    }
    noc_async_write_one_packet_set_state(get_noc_addr(x, y, dst_l1), F);
    for (uint32_t t = 0; t < n; ++t) {
        noc_async_write_one_packet_with_state(src + t * T, dst_l1 + t * T);
        noc_async_write_one_packet_with_state(src + t * T + 2 * F, dst_l1 + t * T + 2 * F);
    }
}

// Word offset of element (r, 0) of a tile (face 0 row r for r < 16, face 2 row r - 16 otherwise).
constexpr uint32_t col0_index(uint32_t r) { return r < FACE_DIM ? r * FACE_DIM : 512 + (r - FACE_DIM) * FACE_DIM; }

// Column 0 of a tile <-> one 32-word fp32 record. Loads go through plain pointers in batches of 8 (fully unrolled,
// constant offsets) so the compiler issues them back to back instead of stalling on every load-use pair; the caller
// has invalidated the L1 cache after the producer's data landed. Stores stay volatile (an unpacker / NoC consumer).
// fp32 interm: word (r, 0) <-> rec[r]; Float16_b interm: the bf16 at (r, 0) is the top half of the fp32 word (exact).
template <bool fp32>
FORCE_INLINE void tile_col0_to_record(uint32_t tile_addr, uint32_t rec_addr) {
    volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rec_addr);
#pragma GCC unroll 4
    for (uint32_t b = 0; b < TILE_DIM; b += 8) {
        uint32_t v[8];
        if constexpr (fp32) {
            const uint32_t* tile = reinterpret_cast<const uint32_t*>(tile_addr);
#pragma GCC unroll 8
            for (uint32_t i = 0; i < 8; ++i) {
                v[i] = tile[col0_index(b + i)];
            }
        } else {
            const uint16_t* tile = reinterpret_cast<const uint16_t*>(tile_addr);
#pragma GCC unroll 8
            for (uint32_t i = 0; i < 8; ++i) {
                v[i] = static_cast<uint32_t>(tile[col0_index(b + i)]) << 16;
            }
        }
#pragma GCC unroll 8
        for (uint32_t i = 0; i < 8; ++i) {
            rec[b + i] = v[i];
        }
    }
}

template <bool fp32>
FORCE_INLINE void record_to_tile_col0(uint32_t rec_addr, uint32_t tile_addr) {
    const uint32_t* rec = reinterpret_cast<const uint32_t*>(rec_addr);
#pragma GCC unroll 4
    for (uint32_t b = 0; b < TILE_DIM; b += 8) {
        uint32_t v[8];
#pragma GCC unroll 8
        for (uint32_t i = 0; i < 8; ++i) {
            v[i] = rec[b + i];
        }
        if constexpr (fp32) {
            volatile tt_l1_ptr uint32_t* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr);
#pragma GCC unroll 8
            for (uint32_t i = 0; i < 8; ++i) {
                tile[col0_index(b + i)] = v[i];
            }
        } else {
            volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_addr);
#pragma GCC unroll 8
            for (uint32_t i = 0; i < 8; ++i) {
                tile[col0_index(b + i)] = static_cast<uint16_t>(v[i] >> 16);
            }
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t C = get_compile_time_arg_val(1);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t WS = get_compile_time_arg_val(4);
    constexpr bool sharded = get_compile_time_arg_val(6) != 0;
    constexpr bool combine = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t G_w = get_compile_time_arg_val(8);
    constexpr uint32_t Rw = get_compile_time_arg_val(9);
    constexpr uint32_t nwpb = get_compile_time_arg_val(10);
    constexpr uint32_t G = get_compile_time_arg_val(11);
    constexpr uint32_t T_in = get_compile_time_arg_val(12);
    constexpr uint32_t T_i = get_compile_time_arg_val(13);
    constexpr uint32_t w_valid_last = get_compile_time_arg_val(14);
    constexpr uint32_t es = get_compile_time_arg_val(15);  // 0: Bfp8_b (never W-pad zeroed: excluded)
    constexpr bool masked_cfg = get_compile_time_arg_val(16) != 0;
    constexpr uint32_t num_waves_total = get_compile_time_arg_val(17);
    constexpr bool interm_fp32 = get_compile_time_arg_val(18) != 0;
    // Wave-ordered reads (host READ_WAVE_GATE): the reads of a wave's rows are issued only once this core's previous
    // wave is reduced locally, so the grid's wave-0 reads get the DRAM first (the exchange starts after the slowest
    // root publishes wave 0) instead of competing with the fast cores' wave-1 reads.
    constexpr bool read_wave_gate = get_compile_time_arg_val(19) != 0;
    constexpr auto input_args = TensorAccessorArgs<20>();
    static_assert(num_w_chunks * C == WS || sharded, "interleaved rows are pushed as whole Wp windows");

    uint32_t a = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(a++);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(a++);
    const uint32_t rows = get_arg_val<uint32_t>(a++);
    const uint32_t w0 = get_arg_val<uint32_t>(a++);
    const uint32_t V = get_arg_val<uint32_t>(a++);
    const bool is_root = get_arg_val<uint32_t>(a++) != 0;
    const uint32_t my_index = get_arg_val<uint32_t>(a++);
    const uint32_t root_x = get_arg_val<uint32_t>(a++);
    const uint32_t root_y = get_arg_val<uint32_t>(a++);
    const uint32_t landing = get_arg_val<uint32_t>(a++);
    const uint32_t seg_stride = get_arg_val<uint32_t>(a++);
    const uint32_t rec_off = get_arg_val<uint32_t>(a++);
    const uint32_t n_send = get_arg_val<uint32_t>(a++);
    const uint32_t send_port[2] = {get_arg_val<uint32_t>(a), get_arg_val<uint32_t>(a + 1)};
    a += 2;
    const uint32_t n_in_fwd = get_arg_val<uint32_t>(a++);
    const uint32_t n_in_bwd = get_arg_val<uint32_t>(a++);
    const uint32_t my_slot = get_arg_val<uint32_t>(a++);
    const uint32_t slot_port_idx = a;  // G packed port coords: where slot q lands on this chip
    a += G;
    const uint32_t entry_slot_idx = a;  // n_in_fwd + n_in_bwd slot ids: entry i of a direction (landing order)
    a += n_in_fwd + n_in_bwd;
    // root only: the members as row-runs (logical rectangles of one row): [n_runs, then per run lo, hi (packed
    // virtual coords), count]
    const uint32_t member_idx = a;

    const auto input_accessor = TensorAccessor(input_args, input_addr, T_in);
    Noc noc;

    // ---- prologue ----
    if constexpr (masked_cfg) {
        dataflow_kernel_lib::calculate_and_prepare_partial_reduce_scalers<
            cb_reduce_scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            w_valid_last>();
    } else {
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            cb_reduce_scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW>();
    }
    CircularBuffer x_cb(cb_x_tiles);
    if constexpr (sharded) {
        x_cb.reserve_back(rows * WS);
        x_cb.push_back(rows * WS);
    }

    CircularBuffer partial_cb(cb_partial_col);
    CircularBuffer gather_cb(cb_gather);
    CircularBuffer rms_root_cb(cb_rms_root);
    CircularBuffer inv_cb(cb_inv_rms);
    CircularBuffer dev_cb(cb_dev_partial);
    CircularBuffer stats_cb(cb_stats_gather);
    // Base addresses every core shares (identical CB layout; a full block returns every write pointer to its base).
    const uint32_t gather_base = combine ? gather_cb.get_write_ptr() : 0;
    const uint32_t inv_base = inv_cb.get_write_ptr();
    const uint32_t staging = CircularBuffer(cb_stats_record).get_write_ptr();  // scratch, never pushed
    const uint32_t own_staging = staging + G * Rw * RECORD_BYTES;

    const uint32_t num_blocks = (rows + R - 1) / R;
    const uint32_t own_waves = num_blocks * nwpb;
    auto wave_n = [&](uint32_t w) -> uint32_t {
        const uint32_t b = w / nwpb;
        if (b >= num_blocks) {
            return 0;
        }
        const uint32_t rem = rows - b * R;
        const uint32_t rb = rem < R ? rem : R;
        const uint32_t first = (w - b * nwpb) * Rw;
        if (first >= rb) {
            return 0;
        }
        return (rb - first) < Rw ? rb - first : Rw;
    };
    auto unpack_x = [](uint32_t v) { return v >> 16; };
    auto unpack_y = [](uint32_t v) { return v & 0xFFFF; };

    // Local reduction progress (read_wave_gate): consumed waves of this core's partial CB (member: sent, combine
    // root: gathered, R1 root: published) and the partial CB's front.
    uint32_t ws = 0, wi = 0, inv_count = 0;  // member cursors (send, receive)
    uint32_t wg = 0;                         // root gather cursor
    uint32_t wp = 0;                         // root publish cursor
    [[maybe_unused]] auto wave_reduced = [&](uint32_t w) -> bool {
        const uint32_t consumed = !is_root ? ws : (combine ? wg : wp);
        if (consumed > w) {
            return true;
        }
        CircularBuffer& src = combine ? partial_cb : dev_cb;
        return consumed == w && src.pages_available_at_front(1);
    };

    // ---- load cursor (interleaved), non-blocking: one tile-row's reads in flight; a later pass pushes it once the
    // reads have landed, so the sends / publishes of the waves already reduced are not held behind a row's DRAM
    // latency (measured: a member's wave-0 partial waited ~1.6 us in the row-1 read barrier at 640x1792). ----
    uint32_t lb = 0, lr = 0;
    bool load_inflight = false;
    auto try_load = [&]() -> bool {
        if constexpr (sharded) {
            return false;
        } else {
            if (load_inflight) {
                if (!ncrisc_noc_reads_flushed(noc_index)) {
                    return false;
                }
                DRMS_ZONE("rd_land");
                invalidate_l1_cache();
                if constexpr (w_valid_last < TILE_DIM && es != 0) {
                    if (w0 + V == Wt) {
                        zero_tile_w_padding<es>(noc, x_cb, (V - 1) * T_in, w_valid_last);
                    }
                }
                x_cb.push_back(WS);
                const uint32_t rem = rows - lb * R;
                const uint32_t rb = rem < R ? rem : R;
                if (++lr == rb) {
                    lr = 0;
                    ++lb;
                }
                load_inflight = false;
                return true;
            }
            if (lb >= num_blocks || !x_cb.pages_reservable_at_back(WS)) {
                return false;
            }
            if constexpr (read_wave_gate) {
                // first row of wave k >= 1 of the core: wait until wave k-1's local partial exists (pushed by compute,
                // or already consumed by this reader)
                const uint32_t row = lb * R + lr;
                if (row > 0 && lr % Rw == 0 && !wave_reduced(row / Rw - 1)) {
                    return false;
                }
            }
            DRMS_ZONE("rd_load");
            const uint32_t tile_row = start_tile_row + lb * R + lr;
            x_cb.reserve_back(WS);
            const uint32_t l1 = x_cb.get_write_ptr();
            const uint32_t row_page = tile_row * Wt + w0;
            for (uint32_t j = 0; j < V; ++j) {
                noc_async_read_tile(row_page + j, input_accessor, l1 + j * T_in);
            }
            load_inflight = true;
            return true;
        }
    };

    // ---- member cursors (R3, index > 0) ----
    auto try_send = [&]() -> bool {
        while (ws < own_waves && wave_n(ws) == 0) {
            ++ws;
        }
        if (ws >= own_waves) {
            return false;
        }
        const uint32_t n = wave_n(ws);
        if (!partial_cb.pages_available_at_front(n)) {
            return false;
        }
        DRMS_ZONE("mb_send");
        const uint32_t k = ws % nwpb;
        write_faces_0_2<T_i>(
            partial_cb.get_read_ptr(), root_x, root_y, gather_base + (k * G_w + my_index) * Rw * T_i, n);
        noc_async_write_barrier();  // the partial has landed before the root can count it
        noc_semaphore_inc(get_noc_addr(root_x, root_y, get_semaphore(SEM_GATHER_BASE + k)), 1);
        partial_cb.pop_front(n);
        ++ws;
        return true;
    };
    auto try_receive = [&]() -> bool {
        while (wi < own_waves && wave_n(wi) == 0) {
            ++wi;
        }
        if (wi >= own_waves) {
            return false;
        }
        const uint32_t n = wave_n(wi);
        if (sem_value(SEM_INV) < inv_count + 1 || !inv_cb.pages_reservable_at_back(n)) {
            return false;
        }
        DRMS_ZONE("mb_recv");
        inv_cb.reserve_back(n);
        inv_cb.push_back(n);
        ++inv_count;
        ++wi;
        return true;
    };

    // ---- root cursors ----
    static_assert(nwpb <= MAX_WAVES_PER_BLOCK, "one gather semaphore per wave of a block");
    uint32_t gather_count[nwpb] = {};  // per wave-in-block: arrivals consumed so far (cumulative over blocks)
    auto try_gather = [&]() -> bool {
        while (wg < own_waves && wave_n(wg) == 0) {
            ++wg;
        }
        if (wg >= own_waves) {
            return false;
        }
        const uint32_t n = wave_n(wg);
        const uint32_t k = wg % nwpb;
        if (!partial_cb.pages_available_at_front(n) || !gather_cb.pages_reservable_at_back(G_w * Rw) ||
            sem_value(SEM_GATHER_BASE + k) < gather_count[k] + (G_w - 1)) {
            return false;
        }
        DRMS_ZONE("rt_gather");
        gather_cb.reserve_back(G_w * Rw);
        noc_async_read(get_noc_addr(partial_cb.get_read_ptr()), gather_cb.get_write_ptr(), n * T_i);
        noc_async_read_barrier();
        gather_cb.push_back(G_w * Rw);
        partial_cb.pop_front(n);
        gather_count[k] += G_w - 1;
        ++wg;
        return true;
    };

    auto try_publish = [&]() -> bool {
        if (wp >= num_waves_total) {
            return false;
        }
        const uint32_t n = wave_n(wp);
        if (n > 0 && !dev_cb.pages_available_at_front(n)) {
            return false;
        }
        if (wp > 0 && sem_value(SEM_OWN_COUNTED) < n_send * wp) {
            return false;
        }
        DRMS_ZONE("rt_publish");
        if (n > 0) {
            // column 0 of each device-partial tile -> 32 contiguous fp32 words (bf16: exact widening)
            const uint32_t src = dev_cb.get_read_ptr();
            invalidate_l1_cache();
            for (uint32_t t = 0; t < n; ++t) {
                tile_col0_to_record<interm_fp32>(src + t * T_i, own_staging + t * RECORD_BYTES);
            }
            const uint32_t dst = landing + (wp * G + my_slot) * seg_stride + rec_off;
            for (uint32_t i = 0; i < n_send; ++i) {
                noc_async_write(
                    own_staging, get_noc_addr(unpack_x(send_port[i]), unpack_y(send_port[i]), dst), n * RECORD_BYTES);
            }
            noc_async_write_barrier();
            dev_cb.pop_front(n);
        }
        for (uint32_t i = 0; i < n_send; ++i) {
            noc_semaphore_inc(
                get_noc_addr(unpack_x(send_port[i]), unpack_y(send_port[i]), get_semaphore(SEM_OWN_STATS)), 1);
        }
        ++wp;
        return true;
    };

    // collect, one slot per step (lamp L4): wave wc's cb_stats_gather window is reserved once the root published wc;
    // the own slot is read at once, every peer slot as soon as its entry is counted on SEM_LANDED_<d> (the ports
    // notify per landed entry, in (wave, entry) order), so at the last landing one slot is left to read and scatter.
    // Entries of waves this root does not collect (empty waves) are never read: the cursors jump past them.
    uint32_t wc = 0;
    bool wc_open = false;
    uint32_t pending = 0;  // slots of wave wc not yet in cb_stats_gather (bit q)
    uint32_t seen[2] = {0, 0};
    const uint32_t n_in[2] = {n_in_fwd, n_in_bwd};
    auto read_slot = [&](uint32_t q, uint32_t n) {
        DRMS_ZONE("rt_collect");
        const uint32_t pp = get_arg_val<uint32_t>(slot_port_idx + q);
        const uint32_t rec = staging + q * Rw * RECORD_BYTES;
        noc_async_read(
            get_noc_addr(unpack_x(pp), unpack_y(pp), landing + (wc * G + q) * seg_stride + rec_off),
            rec,
            n * RECORD_BYTES);
        noc_async_read_barrier();
        invalidate_l1_cache();
        const uint32_t dst = stats_cb.get_write_ptr();
        for (uint32_t t = 0; t < n; ++t) {
            record_to_tile_col0<interm_fp32>(rec + t * RECORD_BYTES, dst + (t * G + q) * T_i);
        }
        pending &= ~(1u << q);
    };
    auto try_collect = [&]() -> bool {
        if (!wc_open) {
            while (wc < own_waves && wave_n(wc) == 0) {
                ++wc;
            }
            if (wc >= own_waves || wc >= wp || !stats_cb.pages_reservable_at_back(G * wave_n(wc))) {
                return false;
            }
            stats_cb.reserve_back(G * wave_n(wc));
            wc_open = true;
            pending = (1u << G) - 1;
            for (uint32_t d = 0; d < 2; ++d) {
                seen[d] = seen[d] > wc * n_in[d] ? seen[d] : wc * n_in[d];
            }
            read_slot(my_slot, wave_n(wc));  // published (and its write barrier passed) before wp moved past wc
        } else {
            bool stepped = false;
            for (uint32_t d = 0; d < 2 && !stepped; ++d) {
                if (seen[d] < (wc + 1) * n_in[d] && sem_value(SEM_LANDED_FWD + d) > seen[d]) {
                    const uint32_t i = seen[d] - wc * n_in[d];
                    read_slot(get_arg_val<uint32_t>(entry_slot_idx + (d == 0 ? 0 : n_in_fwd) + i), wave_n(wc));
                    ++seen[d];
                    stepped = true;
                }
            }
            if (!stepped) {
                return false;
            }
        }
        if (pending == 0) {
            stats_cb.push_back(G * wave_n(wc));
            wc_open = false;
            ++wc;
        }
        return true;
    };

    uint32_t wd = 0;
    uint32_t inv_sent = 0;  // waves distributed (the value set on the members' SEM_INV)
    auto try_distribute = [&]() -> bool {
        while (wd < own_waves && wave_n(wd) == 0) {
            ++wd;
        }
        if (wd >= own_waves) {
            return false;
        }
        const uint32_t n = wave_n(wd);
        if (!rms_root_cb.pages_available_at_front(n) || !inv_cb.pages_reservable_at_back(n)) {
            return false;
        }
        DRMS_ZONE("rt_distribute");
        const uint32_t k = wd % nwpb;
        const uint32_t src = rms_root_cb.get_read_ptr();
        const uint32_t dst = inv_base + k * Rw * T_i;
        // Own copy first: this root's compute (the next wave's finalize) is not held behind the member writes.
        inv_cb.reserve_back(n);
        noc_async_read(get_noc_addr(src), inv_cb.get_write_ptr(), n * T_i);
        noc_async_read_barrier();
        inv_cb.push_back(n);
        // Members: per row-run (a logical rectangle of one row; lamp L6) one multicast of faces 0 / 2 of each tile,
        // LINKED to the multicast set of the cumulative wave count on SEM_INV (the root's own SEM_INV word, unused on
        // a root, holds the value): the linked chain to one rectangle arrives in order, so no ack barrier sits between
        // the data and the flag. The source pages are released after the writes have left L1 (flushed).
        const uint32_t n_runs = get_arg_val<uint32_t>(member_idx);
        constexpr uint32_t F = T_i / 4;
        const uint32_t sem_inv = get_semaphore(SEM_INV);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_inv) = ++inv_sent;
        for (uint32_t r = 0; r < n_runs; ++r) {
            const uint32_t lo = get_arg_val<uint32_t>(member_idx + 1 + 3 * r);
            const uint32_t hi = get_arg_val<uint32_t>(member_idx + 2 + 3 * r);
            const uint32_t cnt = get_arg_val<uint32_t>(member_idx + 3 + 3 * r);
            auto mc = [&](uint32_t l1) -> uint64_t {
                if constexpr (noc_index == 1) {
                    return get_noc_multicast_addr(unpack_x(hi), unpack_y(hi), unpack_x(lo), unpack_y(lo), l1);
                } else {
                    return get_noc_multicast_addr(unpack_x(lo), unpack_y(lo), unpack_x(hi), unpack_y(hi), l1);
                }
            };
            for (uint32_t t = 0; t < n; ++t) {
                for (uint32_t f = 0; f < 4; f += 2) {
                    const uint32_t off = t * T_i + f * F;
                    noc_async_write_multicast(src + off, mc(dst + off), F, cnt, true);
                }
            }
            noc_semaphore_set_multicast(sem_inv, mc(sem_inv), cnt);
        }
        noc_async_writes_flushed();
        rms_root_cb.pop_front(n);
        ++wd;
        return true;
    };

    // ---- the poll loop ----
    while (true) {
        invalidate_l1_cache();
        try_load();
        if (is_root) {
            // Downstream first: a ready distribute / collect of wave w is on the critical chain to the output
            // write, while the next wave's gather / publish only feeds the exchange (which is still in flight).
            if constexpr (combine) {
                try_distribute();
            }
            try_collect();
            try_publish();
            if constexpr (combine) {
                try_gather();
            }
        } else {
            try_send();
            try_receive();
        }
        bool done = sharded || lb >= num_blocks;
        if (is_root) {
            done = done && wp >= num_waves_total && wc >= own_waves;
            if constexpr (combine) {
                done = done && wg >= own_waves && wd >= own_waves;
            }
        } else {
            done = done && ws >= own_waves && wi >= own_waves;
        }
        if (done) {
            break;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
