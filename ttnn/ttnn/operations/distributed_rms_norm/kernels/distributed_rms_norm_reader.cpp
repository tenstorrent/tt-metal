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
// wave's exchange:
//   load        (interleaved) the next tile-row of x: V page reads, one barrier, W-pad columns zeroed, push WS
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
//     collect   every direction's SEM_LANDED >= (w+1) * n_in: read the G slots' record runs (own slot included,
//               from its own port landing: every chip sums identical fp32 inputs in the same order), scatter each
//               record into column 0 of tile t*G + q of cb_stats_gather, push G*n
//     distribute (R3) cb_rms_root -> every member's cb_inv_rms (faces 0/2, one write per member), barrier,
//               member SEM_INV += 1; own copy into cb_inv_rms
// Raw NoC / CB calls: no kernel_lib helper reads TILE pages with W-pad zeroing, moves a face-strided subset of a tile,
// extracts / scatters a tile's column 0, or polls several event sources without blocking (op_design.md -> Helpers
// considered and rejected).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

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
FORCE_INLINE uint32_t col0_index(uint32_t r) { return r < FACE_DIM ? r * FACE_DIM : 512 + (r - FACE_DIM) * FACE_DIM; }

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
    constexpr auto input_args = TensorAccessorArgs<19>();
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
    const uint32_t member_idx = a;  // G_w - 1 packed member coords (root only)

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

    // ---- load cursor (interleaved) ----
    uint32_t lb = 0, lr = 0;
    auto try_load = [&]() -> bool {
        if constexpr (sharded) {
            return false;
        } else {
            if (lb >= num_blocks || !x_cb.pages_reservable_at_back(WS)) {
                return false;
            }
            const uint32_t tile_row = start_tile_row + lb * R + lr;
            x_cb.reserve_back(WS);
            const uint32_t l1 = x_cb.get_write_ptr();
            const uint32_t row_page = tile_row * Wt + w0;
            for (uint32_t j = 0; j < V; ++j) {
                noc_async_read_tile(row_page + j, input_accessor, l1 + j * T_in);
            }
            noc_async_read_barrier();
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
            return true;
        }
    };

    // ---- member cursors (R3, index > 0) ----
    uint32_t ws = 0, wi = 0, inv_count = 0;
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
        inv_cb.reserve_back(n);
        inv_cb.push_back(n);
        ++inv_count;
        ++wi;
        return true;
    };

    // ---- root cursors ----
    static_assert(nwpb <= MAX_WAVES_PER_BLOCK, "one gather semaphore per wave of a block");
    uint32_t wg = 0;
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
        gather_cb.reserve_back(G_w * Rw);
        noc_async_read(get_noc_addr(partial_cb.get_read_ptr()), gather_cb.get_write_ptr(), n * T_i);
        noc_async_read_barrier();
        gather_cb.push_back(G_w * Rw);
        partial_cb.pop_front(n);
        gather_count[k] += G_w - 1;
        ++wg;
        return true;
    };

    uint32_t wp = 0;
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
        if (n > 0) {
            // column 0 of each device-partial tile -> 32 contiguous fp32 words (bf16: exact widening)
            const uint32_t src = dev_cb.get_read_ptr();
            for (uint32_t t = 0; t < n; ++t) {
                volatile tt_l1_ptr uint32_t* rec =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(own_staging + t * RECORD_BYTES);
                if constexpr (interm_fp32) {
                    volatile tt_l1_ptr uint32_t* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + t * T_i);
                    for (uint32_t r = 0; r < TILE_DIM; ++r) {
                        rec[r] = tile[col0_index(r)];
                    }
                } else {
                    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src + t * T_i);
                    for (uint32_t r = 0; r < TILE_DIM; ++r) {
                        rec[r] = static_cast<uint32_t>(tile[col0_index(r)]) << 16;
                    }
                }
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

    uint32_t wc = 0;
    auto try_collect = [&]() -> bool {
        while (wc < own_waves && wave_n(wc) == 0) {
            ++wc;
        }
        if (wc >= own_waves || wc >= wp) {
            return false;
        }
        const uint32_t n = wave_n(wc);
        if ((n_in_fwd > 0 && sem_value(SEM_LANDED_FWD) < (wc + 1) * n_in_fwd) ||
            (n_in_bwd > 0 && sem_value(SEM_LANDED_BWD) < (wc + 1) * n_in_bwd) ||
            !stats_cb.pages_reservable_at_back(G * n)) {
            return false;
        }
        for (uint32_t q = 0; q < G; ++q) {
            const uint32_t pp = get_arg_val<uint32_t>(slot_port_idx + q);
            noc_async_read(
                get_noc_addr(unpack_x(pp), unpack_y(pp), landing + (wc * G + q) * seg_stride + rec_off),
                staging + q * Rw * RECORD_BYTES,
                n * RECORD_BYTES);
        }
        noc_async_read_barrier();
        stats_cb.reserve_back(G * n);
        const uint32_t dst = stats_cb.get_write_ptr();
        for (uint32_t t = 0; t < n; ++t) {
            for (uint32_t q = 0; q < G; ++q) {
                volatile tt_l1_ptr uint32_t* rec =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(staging + (q * Rw + t) * RECORD_BYTES);
                const uint32_t tile_addr = dst + (t * G + q) * T_i;
                if constexpr (interm_fp32) {
                    volatile tt_l1_ptr uint32_t* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr);
                    for (uint32_t r = 0; r < TILE_DIM; ++r) {
                        tile[col0_index(r)] = rec[r];
                    }
                } else {
                    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_addr);
                    for (uint32_t r = 0; r < TILE_DIM; ++r) {
                        tile[col0_index(r)] = static_cast<uint16_t>(rec[r] >> 16);
                    }
                }
            }
        }
        stats_cb.push_back(G * n);
        ++wc;
        return true;
    };

    uint32_t wd = 0;
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
        const uint32_t k = wd % nwpb;
        const uint32_t src = rms_root_cb.get_read_ptr();
        const uint32_t dst = inv_base + k * Rw * T_i;
        for (uint32_t m = 0; m + 1 < G_w; ++m) {
            const uint32_t mm = get_arg_val<uint32_t>(member_idx + m);
            write_faces_0_2<T_i>(src, unpack_x(mm), unpack_y(mm), dst, n);
        }
        inv_cb.reserve_back(n);
        noc_async_read(get_noc_addr(src), inv_cb.get_write_ptr(), n * T_i);
        noc_async_read_barrier();
        noc_async_write_barrier();
        for (uint32_t m = 0; m + 1 < G_w; ++m) {
            const uint32_t mm = get_arg_val<uint32_t>(member_idx + m);
            noc_semaphore_inc(get_noc_addr(unpack_x(mm), unpack_y(mm), get_semaphore(SEM_INV)), 1);
        }
        inv_cb.push_back(n);
        rms_root_cb.pop_front(n);
        ++wd;
        return true;
    };

    // ---- the poll loop ----
    while (true) {
        invalidate_l1_cache();
        if (is_root) {
            if constexpr (combine) {
                try_gather();
            }
            try_publish();
            try_collect();
            if constexpr (combine) {
                try_distribute();
            }
        } else {
            try_send();
            try_receive();
        }
        try_load();
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
