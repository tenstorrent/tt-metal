// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// distributed_rms_norm reader (NCRISC, NoC0) on every worker core (op_design.md -> Block schedule).
// Passes (Refinement 1): Phase A + Phase B below run once per pass block of PW waves (one pass in Phase 0). Before a
// later pass writes into other members' cb_gather regions / cb_inv_rms slots, a row-group barrier (SEM_PASS) makes
// sure every member's compute is done with the previous pass; SEM_GATHER fields are cleared at that barrier.
//
// Prologue: reduce scaler (1.0; [full, partial] on the sharded W-pad mask path), cb_stats pages zeroed once (only
// column 0 is ever rewritten).
// Phase A, per wave w (look-ahead 1: the reads of wave w+1 are issued before wave w is serviced):
//   load                 interleaved: wave-ordered reads, 1 + LOOKAHEAD waves in flight (transaction id w + 1; x is
//                        resident), each wave pushed once its own reads landed; pad tiles [V, WS) and the W-pad columns
//                        of tile Wt-1 zeroed; sharded: the resident shard is pushed once (zero-copy)
//   contribute_wave      (G_W > 1) this slice's collapsed partials of owner k's rows -> owner k's gather region
//                        (faces 0/2, base combine transport), then one SEM_GATHER increment per owner
//   (owner) gather push  once every member's partial landed
//   publish_wave (owner) OVERLAP: the chip partials' column 0 -> compact fp32 records (128 B per tile-row), written
//                        into the own slot of both ports of the publisher's link; own_stats[w] increment on each
// Phase B, a non-blocking poll over three in-order queues (no wave waits behind another wave's exchange):
//   collect_wave (owner) once the link ports' landed notifies cover the wave: read the G records of each owned row
//                        from the ports, scatter them into column 0 of cb_stats pages (group-position order)
//   allgather_inv_wave   (owner) the finalized inv tiles -> every member's cb_inv_rms rows; SEM_INV increment each
//   (every member)       once SEM_INV covers the wave's owners, push the wave's Q_W cb_inv_rms pages
//
// All NoC atomics are issued from this RISC-V (NCRISC); the kernel ends with noc_async_atomic_barrier().

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// Measurement-only stage zones (host DRMS_PERF_ZONES=1 -> define DRMS_PERF_ZONES; off in production).
#if defined(DRMS_PERF_ZONES) && DRMS_PERF_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#define DRMS_ZONE(name) DeviceZoneScopedN(name)
#else
#define DRMS_ZONE(name)
#endif

// Measurement-only (host DRMS_ABLATE=exchange; 0 in production): the ports do nothing, so the counted gate and the
// landed waits are skipped (the collected records are stale landing bytes — timing only).
#ifndef DRMS_ABLATE_EXCHANGE
#define DRMS_ABLATE_EXCHANGE 0
#endif

namespace {

constexpr uint32_t cb_x_tiles = 1;
constexpr uint32_t cb_reduce_scaler = 2;
constexpr uint32_t cb_inv_rms = 4;
constexpr uint32_t cb_partial_col = 9;
constexpr uint32_t cb_gather = 10;
constexpr uint32_t cb_rms_root = 11;
constexpr uint32_t cb_chip_partial = 14;
constexpr uint32_t cb_stats = 15;
constexpr uint32_t cb_record_scratch = 17;

constexpr uint32_t FACE_DIM = 16;
constexpr uint32_t RECORD_BYTES = 128;
constexpr uint32_t FWD = 0, BWD = 1, OWN = 2;

// word / element index of column-0 row r (0..31) of a tile: face 0 rows 0-15, face 2 rows 16-31
FORCE_INLINE uint32_t col0_index(uint32_t r) { return r < 16 ? r * 16 : 512 + (r - 16) * 16; }

// Faces 0 and 2 of n tiles (column-0 data: the only part any consumer reads) — base rms_norm_combine.hpp.
template <uint32_t T>
FORCE_INLINE void combine_write_tiles(uint32_t src, uint32_t x, uint32_t y, uint32_t dst_l1, uint32_t n) {
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

FORCE_INLINE uint32_t rotated(uint32_t i, uint32_t my_index, uint32_t n) {
    uint32_t j = my_index + 1 + i;
    while (j >= n) {
        j -= n;
    }
    return j;
}

template <uint32_t es>
FORCE_INLINE void store_zeros(uint32_t addr, uint32_t n_elems) {
    if constexpr (es == 4) {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
        for (uint32_t i = 0; i < n_elems; ++i) {
            p[i] = 0;
        }
    } else if constexpr (es == 2) {
        volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr);
        for (uint32_t i = 0; i < n_elems; ++i) {
            p[i] = 0;
        }
    } else {
        volatile tt_l1_ptr uint8_t* p = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(addr);
        for (uint32_t i = 0; i < n_elems; ++i) {
            p[i] = 0;
        }
    }
}

// Zero columns [w_valid, 32) of a tile at tile_offset of the CB's write window (base zero_tile_w_padding).
// es = 4 / 2: Float32 / Float16_b datums. es = 1: Bfp8_b — the 1 B sign+magnitude mantissas (64 B exponent section
// first, then 4 faces of 16x16 bytes); a zero magnitude is zero whatever the shared exponent, so the exponent bytes
// are left alone. Whole pad faces go through the NoC zero loopback, partial face rows through element stores.
template <uint32_t es>
FORCE_INLINE void zero_tile_w_padding(
    const Noc& noc, const CircularBuffer& cb, uint32_t tile_offset, uint32_t w_valid) {
    constexpr uint32_t head = es == 1 ? 64 : 0;
    constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * es;
    constexpr uint32_t face_row_bytes = FACE_DIM * es;
    const uint32_t tile_addr = cb.get_write_ptr() + tile_offset + head;
    if (w_valid <= FACE_DIM) {
        noc.async_write_zeros(cb, face_bytes, {.offset_bytes = tile_offset + head + 1 * face_bytes});
        noc.async_write_zeros(cb, face_bytes, {.offset_bytes = tile_offset + head + 3 * face_bytes});
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

}  // namespace

void kernel_main() {
    constexpr bool SHARDED = get_compile_time_arg_val(0) != 0;
    constexpr uint32_t G_W = get_compile_time_arg_val(1);
    constexpr uint32_t WS = get_compile_time_arg_val(2);
    constexpr uint32_t NW = get_compile_time_arg_val(3);
    constexpr uint32_t Q_W = get_compile_time_arg_val(4);
    constexpr uint32_t OWNERS = get_compile_time_arg_val(5);
    constexpr uint32_t Q = get_compile_time_arg_val(6);
    constexpr uint32_t G = get_compile_time_arg_val(7);
    constexpr uint32_t L = get_compile_time_arg_val(8);
    constexpr uint32_t T_IN = get_compile_time_arg_val(9);
    constexpr uint32_t T_I = get_compile_time_arg_val(10);
    constexpr bool INTERM_FP32 = get_compile_time_arg_val(11) != 0;
    constexpr bool MASKED = get_compile_time_arg_val(12) != 0;
    constexpr uint32_t W_VALID_LAST = get_compile_time_arg_val(13);
    constexpr uint32_t DTYPE_KIND = get_compile_time_arg_val(14);  // 0 fp32, 1 bf16, 2 bfp8_b
    constexpr uint32_t SEM_GATHER = get_compile_time_arg_val(15);
    constexpr uint32_t SEM_INV = get_compile_time_arg_val(16);
    // SEM_GATHER holds one GB-bit field per owned wave (ordinal o): a contributor running a wave ahead can never
    // complete an earlier owned wave's count.
    constexpr uint32_t GB = get_compile_time_arg_val(22);
    constexpr uint32_t SEM_LANDED_BASE = get_compile_time_arg_val(17);
    constexpr uint32_t SEM_OWN_STATS_BASE = get_compile_time_arg_val(18);  // + wave, on the port cores
    constexpr uint32_t SEG_STRIDE = get_compile_time_arg_val(20);
    constexpr uint32_t WT = get_compile_time_arg_val(21);
    constexpr uint32_t LOOKAHEAD = get_compile_time_arg_val(19);  // waves of x reads in flight beyond the oldest
    // passes: the core's waves run in pass blocks of PW waves (Phase A + Phase B each); cb_gather holds REGIONS
    // owned-wave regions and cb_inv_rms PW waves, reused pass after pass behind a row-group barrier (SEM_PASS)
    constexpr uint32_t PW = get_compile_time_arg_val(23);
    constexpr uint32_t REGIONS = get_compile_time_arg_val(24);
    constexpr uint32_t SEM_PASS = get_compile_time_arg_val(25);
    constexpr uint32_t NP = (NW + PW - 1) / PW;
    // x re-read (L1 fallback, interleaved only): x is never resident — Phase A streams it one tile-row at a time
    // through cb_x_tiles (square + collapse), Phase B's writer re-reads it (cb_x_reread)
    constexpr bool REREAD = get_compile_time_arg_val(26) != 0;
    constexpr auto input_args = TensorAccessorArgs<27>();

    size_t a = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(a++);
    const uint32_t start_row = get_arg_val<uint32_t>(a++);
    const uint32_t rows = get_arg_val<uint32_t>(a++);
    const uint32_t w0 = get_arg_val<uint32_t>(a++);
    const uint32_t V = get_arg_val<uint32_t>(a++);
    const uint32_t my_j = get_arg_val<uint32_t>(a++);
    const uint32_t my_pos = get_arg_val<uint32_t>(a++);
    const uint32_t landing = get_arg_val<uint32_t>(a++);
    const uint32_t group = get_arg_val<uint32_t>(a++);
    const uint32_t n_in_fwd = get_arg_val<uint32_t>(a++);
    const uint32_t n_in_bwd = get_arg_val<uint32_t>(a++);
    const bool holds_pad = get_arg_val<uint32_t>(a++) != 0;
    [[maybe_unused]] const bool masked_core = get_arg_val<uint32_t>(a++) != 0;
    [[maybe_unused]] const uint32_t n_active = get_arg_val<uint32_t>(a++);  // ports per link sending the own slot
    const uint32_t member_idx = a;
    a += G_W;
    const uint32_t port_idx = a;
    a += 2 * L;
    const bool sends_own_fwd = get_arg_val<uint32_t>(a++) != 0;
    const bool sends_own_bwd = get_arg_val<uint32_t>(a++) != 0;
    const uint32_t srcdir_idx = a;
    a += G;

    auto member_x = [&](uint32_t m) { return get_arg_val<uint32_t>(member_idx + m) >> 16; };
    auto member_y = [&](uint32_t m) { return get_arg_val<uint32_t>(member_idx + m) & 0xFFFF; };
    auto port_noc = [&](uint32_t l, uint32_t d, uint32_t addr) {
        const uint32_t v = get_arg_val<uint32_t>(port_idx + 2 * l + d);
        return get_noc_addr(v >> 16, v & 0xFFFF, addr);
    };

    auto n_wave = [&](uint32_t w) -> uint32_t {
        const uint32_t r0 = w * Q_W;
        return r0 >= rows ? 0 : ((rows - r0) < Q_W ? rows - r0 : Q_W);
    };
    auto member_of = [](uint32_t w, uint32_t k) -> uint32_t { return (w * OWNERS + k) % G_W; };
    auto my_k = [&](uint32_t w) -> uint32_t {
        for (uint32_t k = 0; k < OWNERS; ++k) {
            if (member_of(w, k) == my_j) {
                return k;
            }
        }
        return OWNERS;
    };
    auto n_own = [&](uint32_t w, uint32_t k) -> uint32_t {
        const uint32_t n = n_wave(w);
        const uint32_t first = k * Q;
        return first >= n ? 0 : ((n - first) < Q ? n - first : Q);
    };
    // owned-wave ordinal of member m at wave w: how many earlier waves m owned with rows. Its gather region (and
    // SEM_GATHER field) is ordinal mod REGIONS: one pass holds at most REGIONS owned waves, and the ring of regions
    // follows the owner's cb_gather pushes (one region per owned wave with rows).
    auto ordinal = [&](uint32_t m, uint32_t w) -> uint32_t {
        uint32_t o = 0;
        for (uint32_t v = 0; v < w; ++v) {
            for (uint32_t k = 0; k < OWNERS; ++k) {
                if (member_of(v, k) == m && n_own(v, k) > 0) {
                    ++o;
                }
            }
        }
        return o % REGIONS;
    };

    Noc noc;
    CircularBuffer x_cb(cb_x_tiles);

    uint32_t inv_base = 0;  // cb_inv_rms write pointer at the pass start (the same on every member: same pushes)
    [[maybe_unused]] uint32_t gather_base = 0;
    if constexpr (G_W > 1) {
        gather_base = get_write_ptr(cb_gather);
    }
    const uint32_t scratch = get_write_ptr(cb_record_scratch);  // [0, Q) publish staging, [Q, Q + G*Q) collect

    // ---------------- load: wave-ordered reads with a bounded look-ahead (x resident per pass block) ----------------
    // Wave w's reads carry transaction id w + 1 (mod 15: ids 1..15). At most 1 + LOOKAHEAD waves are in flight per
    // core, so DRAM serves the grid's reads roughly in wave order: every row-group's wave 0 lands first and its
    // exchange starts early, instead of competing with every core's last wave.
    [[maybe_unused]] uint32_t x_base = 0;
    [[maybe_unused]] uint32_t x_row0 = 0;
    [[maybe_unused]] const auto input_accessor = TensorAccessor(input_args, in_addr);
    uint32_t issued = 0;
    uint32_t issue_end = 0;
    auto trid = [](uint32_t w) -> uint32_t { return 1 + w % 15; };
    auto issue_wave = [&](uint32_t w) {
        if constexpr (!SHARDED) {
            DRMS_ZONE("rd_issue");
            noc_async_read_set_trid(trid(w));
            const uint32_t r0 = w * Q_W;
            const uint32_t n = n_wave(w);
            for (uint32_t r = r0; r < r0 + n; ++r) {
                const uint32_t page0 = (start_row + r) * WT + w0;
                const uint32_t l1 = x_base + (r - x_row0) * WS * T_IN;
                for (uint32_t t = 0; t < V; ++t) {
                    noc_async_read_tile(page0 + t, input_accessor, l1 + t * T_IN);
                }
            }
            noc_async_read_set_trid(0);
        }
    };
    auto issue_ahead = [&](uint32_t upto) {
        while (issued < issue_end && issued <= upto) {
            issue_wave(issued++);
        }
    };
    // reserve + start the x reads of the pass block [w_begin, w_end)
    auto load_pass = [&](uint32_t w_begin, uint32_t w_end) {
        if constexpr (!SHARDED && !REREAD) {
            const uint32_t row0 = w_begin * Q_W;
            const uint32_t row_end = (w_end * Q_W < rows) ? w_end * Q_W : rows;
            const uint32_t rows_p = row_end > row0 ? row_end - row0 : 0;
            issued = w_begin;
            issue_end = w_end;
            if (rows_p == 0) {
                return;
            }
            cb_reserve_back(cb_x_tiles, rows_p * WS);
            x_base = get_write_ptr(cb_x_tiles);
            x_row0 = row0;
            if (V < WS) {
                for (uint32_t r = 0; r < rows_p; ++r) {
                    noc.async_write_zeros(x_cb, (WS - V) * T_IN, {.offset_bytes = (r * WS + V) * T_IN});
                }
                noc.write_zeros_l1_barrier();
            }
            issue_ahead(w_begin + LOOKAHEAD);
        }
    };
    if constexpr (!SHARDED) {
        load_pass(0, PW < NW ? PW : NW);
    } else {
        cb_reserve_back(cb_x_tiles, rows * WS);
        cb_push_back(cb_x_tiles, rows * WS);
    }

    // ---------------- prologue (behind the x reads) ----------------
    {
        DRMS_ZONE("rd_prologue");
        if constexpr (MASKED) {
            dataflow_kernel_lib::calculate_and_prepare_partial_reduce_scalers<
                cb_reduce_scaler,
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                (MASKED ? W_VALID_LAST : 1)>();
        } else {
            dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                cb_reduce_scaler,
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW>();
        }
    }

    // x re-read: the wave's tile-rows streamed one at a time (cb_x_tiles holds a few rows; compute pops each row
    // after its square + collapse). Pad tiles [V, WS) and the W-pad columns are zeroed per row as in the resident path.
    [[maybe_unused]] auto stream_wave = [&](uint32_t w) {
        if constexpr (!SHARDED && REREAD) {
            DRMS_ZONE("rd_stream");
            const uint32_t r0 = w * Q_W;
            const uint32_t n = n_wave(w);
            for (uint32_t r = r0; r < r0 + n; ++r) {
                cb_reserve_back(cb_x_tiles, WS);
                const uint32_t l1 = get_write_ptr(cb_x_tiles);
                if (V < WS) {
                    noc.async_write_zeros(x_cb, (WS - V) * T_IN, {.offset_bytes = V * T_IN});
                }
                const uint32_t page0 = (start_row + r) * WT + w0;
                for (uint32_t t = 0; t < V; ++t) {
                    noc_async_read_tile(page0 + t, input_accessor, l1 + t * T_IN);
                }
                noc_async_read_barrier();
                if (V < WS) {
                    noc.write_zeros_l1_barrier();
                }
                if constexpr (W_VALID_LAST < 32) {
                    if (holds_pad) {
                        constexpr uint32_t es = DTYPE_KIND == 0 ? 4 : (DTYPE_KIND == 1 ? 2 : 1);
                        zero_tile_w_padding<es>(noc, x_cb, (V - 1) * T_IN, W_VALID_LAST);
                    }
                }
                cb_push_back(cb_x_tiles, WS);
            }
        }
    };

    auto land_wave = [&](uint32_t w) {
        if constexpr (REREAD) {
            stream_wave(w);
        } else if constexpr (!SHARDED) {
            DRMS_ZONE("rd_land");
            noc_async_read_barrier_with_trid(trid(w));
            const uint32_t n = n_wave(w);
            if constexpr (W_VALID_LAST < 32) {
                if (holds_pad) {
                    constexpr uint32_t es = DTYPE_KIND == 0 ? 4 : (DTYPE_KIND == 1 ? 2 : 1);
                    // zero_tile_w_padding addresses relative to the CB write pointer, which sits at x_base
                    // offsets are relative to the CB write pointer, which sits at the wave's first row
                    for (uint32_t r = 0; r < n; ++r) {
                        zero_tile_w_padding<es>(noc, x_cb, (r * WS + V - 1) * T_IN, W_VALID_LAST);
                    }
                }
            }
            cb_push_back(cb_x_tiles, n * WS);
        }
    };

    // ---------------- service of wave w (Phase A): contribute, gather, publish ----------------
    auto publish = [&](uint32_t w, uint32_t k) {
        const uint32_t no = n_own(w, k);
        const uint32_t kp = group * OWNERS + k;
        const uint32_t l = kp % L;
        const uint32_t kl = kp / L;
        volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        if (no > 0) {
            {
                DRMS_ZONE("rd_wait_chip");
                cb_wait_front(cb_chip_partial, no);
            }
            const uint32_t src = get_read_ptr(cb_chip_partial);
            for (uint32_t r = 0; r < no; ++r) {
                const uint32_t tile = src + r * T_I;
                for (uint32_t i = 0; i < 32; ++i) {
                    if constexpr (INTERM_FP32) {
                        rec[r * 32 + i] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile)[col0_index(i)];
                    } else {
                        rec[r * 32 + i] =
                            static_cast<uint32_t>(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile)[col0_index(i)])
                            << 16;
                    }
                }
            }
            cb_pop_front(cb_chip_partial, no);
        }
        for (uint32_t r = no; r < Q; ++r) {
            for (uint32_t i = 0; i < 32; ++i) {
                rec[r * 32 + i] = 0;
            }
        }
        DRMS_ZONE("rd_publish");
        const uint32_t off = landing + (w * G + my_pos) * SEG_STRIDE + kl * Q * RECORD_BYTES;
        if (sends_own_fwd) {
            noc_async_write(scratch, port_noc(l, FWD, off), Q * RECORD_BYTES);
        }
        if (sends_own_bwd) {
            noc_async_write(scratch, port_noc(l, BWD, off), Q * RECORD_BYTES);
        }
        noc_async_write_barrier();
        // announce wave w on the port's own_stats[w] (one semaphore per wave: no ordering gate needed)
        const uint32_t own_stats = get_semaphore(SEM_OWN_STATS_BASE + w);
        if (sends_own_fwd) {
            noc_semaphore_inc(port_noc(l, FWD, own_stats), 1);
        }
        if (sends_own_bwd) {
            noc_semaphore_inc(port_noc(l, BWD, own_stats), 1);
        }
    };

    auto service = [&](uint32_t w) {
        const uint32_t n = n_wave(w);
        if constexpr (G_W > 1) {
            if (n > 0) {
                {
                    DRMS_ZONE("rd_wait_partial");
                    cb_wait_front(cb_partial_col, n);
                }
                DRMS_ZONE("rd_contrib");
                const uint32_t src = get_read_ptr(cb_partial_col);
                const uint32_t gsem = get_semaphore(SEM_GATHER);
                for (uint32_t i = 0; i < OWNERS; ++i) {
                    const uint32_t k = rotated(i, my_j % OWNERS, OWNERS);
                    const uint32_t no = n_own(w, k);
                    if (no == 0) {
                        continue;
                    }
                    const uint32_t m = member_of(w, k);
                    const uint32_t dst = gather_base + ((ordinal(m, w) * G_W + my_j) * Q) * T_I;
                    combine_write_tiles<T_I>(src + k * Q * T_I, member_x(m), member_y(m), dst, no);
                }
                noc_async_write_barrier();
                for (uint32_t k = 0; k < OWNERS; ++k) {
                    if (n_own(w, k) > 0) {
                        const uint32_t m = member_of(w, k);
                        noc_semaphore_inc(get_noc_addr(member_x(m), member_y(m), gsem), 1u << (GB * ordinal(m, w)));
                    }
                }
                cb_pop_front(cb_partial_col, n);
            }
        }
        const uint32_t k = my_k(w);
        if (k < OWNERS) {
            if constexpr (G_W > 1) {
                if (n_own(w, k) > 0) {
                    const uint32_t field = GB * ordinal(my_j, w);
                    DRMS_ZONE("rd_gather_wait");
                    volatile tt_l1_ptr uint32_t* gs =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GATHER));
                    do {
                        invalidate_l1_cache();
                    } while (((*gs >> field) & ((1u << GB) - 1)) < G_W);
                    cb_reserve_back(cb_gather, G_W * Q);
                    cb_push_back(cb_gather, G_W * Q);
                }
            }
            publish(w, k);
        }
    };

    // ================= Phase B =================
    // A non-blocking poll over three in-order queues, so no wave waits behind another wave's exchange:
    //   collect  (owned waves with rows)  landed notifies satisfied and cb_stats free -> records -> cb_stats
    //   allgather (owned waves)           the finalized rows in cb_rms_root -> every member's cb_inv_rms
    //   inv      (every wave)             SEM_INV reached the wave's owners -> push the wave's Q_W pages
    const uint32_t inv_sem = get_semaphore(SEM_INV);
    uint32_t w_lim = NW;  // end of the current pass block (the Phase B queues stop there)
    auto next_owned = [&](uint32_t from) -> uint32_t {
        uint32_t w = from;
        while (w < w_lim && my_k(w) == OWNERS) {
            ++w;
        }
        return w;
    };
    auto next_owned_rows = [&](uint32_t from) -> uint32_t {
        uint32_t w = next_owned(from);
        while (w < w_lim && n_own(w, my_k(w)) == 0) {
            w = next_owned(w + 1);
        }
        return w;
    };
    auto landed_ok = [&](uint32_t w, uint32_t l) -> bool {
        if constexpr (DRMS_ABLATE_EXCHANGE) {
            return true;
        }
        invalidate_l1_cache();
        if (n_in_fwd > 0 && *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                get_semaphore(SEM_LANDED_BASE + 2 * l + FWD)) < (w + 1) * n_in_fwd) {
            return false;
        }
        if (n_in_bwd > 0 && *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                get_semaphore(SEM_LANDED_BASE + 2 * l + BWD)) < (w + 1) * n_in_bwd) {
            return false;
        }
        return true;
    };
    auto collect = [&](uint32_t w) {
        DRMS_ZONE("rd_collect");
        const uint32_t k = my_k(w);
        const uint32_t no = n_own(w, k);
        const uint32_t kp = group * OWNERS + k;
        const uint32_t l = kp % L;
        const uint32_t kl = kp / L;
        const uint32_t col = scratch + Q * RECORD_BYTES;
        for (uint32_t pos = 0; pos < G; ++pos) {
            uint32_t d = get_arg_val<uint32_t>(srcdir_idx + pos);
            if (d == OWN) {
                d = sends_own_fwd ? FWD : BWD;
            }
            const uint32_t off = landing + (w * G + pos) * SEG_STRIDE + kl * Q * RECORD_BYTES;
            noc_async_read(port_noc(l, d, off), col + pos * Q * RECORD_BYTES, no * RECORD_BYTES);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        cb_reserve_back(cb_stats, G * Q);
        // column-0 scatter: record (pos, r) element i -> page (pos * Q + r), row i of column 0
        uint32_t* st = reinterpret_cast<uint32_t*>(get_write_ptr(cb_stats));
        const uint32_t* recs = reinterpret_cast<const uint32_t*>(col);
        for (uint32_t pr = 0; pr < G * Q; ++pr) {
            const uint32_t r = pr % Q;
            if (r >= no) {
                continue;
            }
            uint32_t* page = st + pr * 1024;
            const uint32_t* rec = recs + pr * 32;
#pragma GCC unroll 16
            for (uint32_t i = 0; i < 16; ++i) {
                page[i * 16] = rec[i];
                page[512 + i * 16] = rec[16 + i];
            }
        }
        asm volatile("" ::: "memory");  // the scatter stores are issued before the push
        cb_push_back(cb_stats, G * Q);
    };
    auto allgather = [&](uint32_t w) {
        DRMS_ZONE("rd_allgather");
        const uint32_t k = my_k(w);
        const uint32_t no = n_own(w, k);
        if (no > 0) {
            const uint32_t src = get_read_ptr(cb_rms_root);
            const uint32_t dst = inv_base + ((w % PW) * Q_W + k * Q) * T_I;
            for (uint32_t i = 0; i < G_W; ++i) {
                const uint32_t m = rotated(i, my_j, G_W);
                combine_write_tiles<T_I>(src, member_x(m), member_y(m), dst, no);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_rms_root, no);
        }
    };
    // SEM_INV counts owner increments cumulatively over waves, so they must reach every member in wave order: the
    // owner of wave w increments only once its own SEM_INV shows every earlier wave complete (>= w * OWNERS). Owners
    // write their rows, barrier, then increment — so any increment observed from wave v implies every wave < v landed
    // on every member.
    auto announce = [&](uint32_t /*w*/) {
        for (uint32_t i = 0; i < G_W; ++i) {
            const uint32_t m = rotated(i, my_j, G_W);
            noc_semaphore_inc(get_noc_addr(member_x(m), member_y(m), inv_sem), 1);
        }
    };
    volatile tt_l1_ptr uint32_t* my_inv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(inv_sem);
    // Row-group pass barrier (passes > 1 only): before pass p writes into other members' cb_gather / cb_inv_rms
    // slots (the same slots pass p-1 used), every member's compute must be done with pass p-1. Each reader learns
    // that of its own compute from cb_inv_rms (every page it pushed was popped), clears its SEM_GATHER fields (all of
    // pass p-1's contributions were counted), then counts itself done on every member's SEM_PASS.
    const uint32_t pass_sem = get_semaphore(SEM_PASS);
    auto pass_barrier = [&](uint32_t p) {
        DRMS_ZONE("rd_pass_barrier");
        while (!cb_pages_reservable_at_back(cb_inv_rms, PW * Q_W)) {
        }
        if constexpr (G_W > 1) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GATHER)) = 0;
        }
        for (uint32_t i = 0; i < G_W; ++i) {
            const uint32_t m = rotated(i, my_j, G_W);
            noc_semaphore_inc(get_noc_addr(member_x(m), member_y(m), pass_sem), 1);
        }
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pass_sem), p * G_W);
    };

    for (uint32_t p = 0; p < NP; ++p) {
        const uint32_t w_begin = p * PW;
        const uint32_t w_end = (w_begin + PW < NW) ? w_begin + PW : NW;
        w_lim = w_end;
        if (p > 0) {
            load_pass(w_begin, w_end);  // x reads of this pass go out before the barrier wait
            pass_barrier(p);
        }

        // ================= Phase A =================
        for (uint32_t w = w_begin; w <= w_end; ++w) {
            if (w < w_end) {
                land_wave(w);
                if constexpr (!REREAD) {
                    issue_ahead(w + 1 + LOOKAHEAD);
                }
            }
            if (w > w_begin) {
                service(w - 1);
            }
        }
        if (p == 0) {
            CircularBuffer stats_cb(cb_stats);
            noc.async_write_zeros(stats_cb, G * Q * 4096, {.offset_bytes = 0});
            noc.write_zeros_l1_barrier();
        }

        // ================= Phase B (the pass's waves) =================
        uint32_t wc = next_owned_rows(w_begin);  // next owned wave (with rows) to collect
        uint32_t wa = next_owned(w_begin);       // next owned wave whose rows to all-gather
        uint32_t wn = next_owned(w_begin);       // next owned wave to announce (SEM_INV increments, wave-ordered)
        uint32_t wi = w_begin;                   // next wave whose inv rows to push
        cb_reserve_back(cb_inv_rms, (w_end - w_begin) * Q_W);
        inv_base = get_write_ptr(cb_inv_rms);
        while (wi < w_end || wn < w_end) {
            if (wc < w_end) {
                const uint32_t kp = group * OWNERS + my_k(wc);
                if (landed_ok(wc, kp % L) && cb_pages_reservable_at_back(cb_stats, G * Q)) {
                    collect(wc);
                    wc = next_owned_rows(wc + 1);
                    continue;
                }
            }
            if (wa < w_end) {
                const uint32_t no = n_own(wa, my_k(wa));
                // a wave with rows is all-gathered only after it was collected (wc moved past it)
                if (no == 0 || (wc > wa && cb_pages_available_at_front(cb_rms_root, no))) {
                    allgather(wa);
                    wa = next_owned(wa + 1);
                    continue;
                }
            }
            invalidate_l1_cache();
            if (wn < wa && *my_inv >= wn * OWNERS) {
                announce(wn);
                wn = next_owned(wn + 1);
                continue;
            }
            if (wi < w_end && *my_inv >= (wi + 1) * OWNERS) {
                cb_push_back(cb_inv_rms, Q_W);
                ++wi;
                continue;
            }
        }
    }
    noc_async_atomic_barrier();
}
