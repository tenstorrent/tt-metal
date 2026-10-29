// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// dedicated_owners experiment: the OWNER reader (NCRISC, NoC0) of one row-group, on a worker core that reads no x.
// It owns every (wave w, owner slot k) of its group that has rows ("slots", in (w, k) order) — the role the baseline
// rotates over the group's x-reading members. One non-blocking poll over four in-order queues:
//   gather    the members' partials of slot s landed (SEM_GATHER field s counts G_W) -> push its cb_gather region
//   publish   compute's chip partial of slot s -> compact fp32 records -> the slot's link ports + own_stats[w]
//   collect   the link ports' landed notifies cover wave w -> the G records of each row into row 0 of cb_stats
//   allgather the finalized rows (cb_rms_root) -> one posted 128 B record per row into every member's cb_inv_rms
// Wire, addresses, records and fold order are the baseline's (distributed_rms_norm_reader.cpp): the members'
// cb_inv_rms / cb_gather and this core's CBs come from one CB descriptor set over the union of the cores, so the
// addresses match. One pass only (the host gates the dedicated path to NP == 1).
// No kernel_lib helper is bypassed beyond what the baseline reader already does (same dataflow primitives).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

#if defined(DRMS_PERF_ZONES) && DRMS_PERF_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#define DRMS_ZONE(name) DeviceZoneScopedN(name)
#else
#define DRMS_ZONE(name)
#endif

// DRMS_OWNER_PUBLISH_ONLY: Phase A only (gather, publish); the members keep the baseline's rotating Phase B owners
#ifndef DRMS_OWNER_PUBLISH_ONLY
#define DRMS_OWNER_PUBLISH_ONLY 0
#endif
#ifndef DRMS_ABLATE_EXCHANGE
#define DRMS_ABLATE_EXCHANGE 0
#endif

namespace {
constexpr uint32_t cb_reduce_scaler = 2;
constexpr uint32_t cb_inv_rms = 4;
constexpr uint32_t cb_gather = 10;
constexpr uint32_t cb_rms_root = 11;
constexpr uint32_t cb_chip_partial = 14;
constexpr uint32_t cb_stats = 15;
constexpr uint32_t cb_record_scratch = 17;

constexpr uint32_t RECORD_BYTES = 128;
constexpr uint32_t STATS_TILE_BYTES = 4096;
constexpr uint32_t STATS_FACE_BYTES = 1024;
constexpr uint32_t FWD = 0, BWD = 1, OWN = 2;
constexpr uint32_t MAX_SLOTS = 32;

FORCE_INLINE uint32_t col0_index(uint32_t r) { return r < 16 ? r * 16 : 512 + (r - 16) * 16; }
}  // namespace

void kernel_main() {
    constexpr uint32_t G_W = get_compile_time_arg_val(1);
    constexpr uint32_t NW = get_compile_time_arg_val(3);
    constexpr uint32_t Q_W = get_compile_time_arg_val(4);
    constexpr uint32_t OWNERS = get_compile_time_arg_val(5);
    constexpr uint32_t Q = get_compile_time_arg_val(6);
    constexpr uint32_t G = get_compile_time_arg_val(7);
    constexpr uint32_t L = get_compile_time_arg_val(8);
    constexpr uint32_t T_I = get_compile_time_arg_val(10);
    constexpr bool INTERM_FP32 = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t SEM_GATHER = get_compile_time_arg_val(15);
    constexpr uint32_t SEM_INV = get_compile_time_arg_val(16);  // publish-only: "own record published" on the B owner
    constexpr uint32_t SEM_LANDED_BASE = get_compile_time_arg_val(17);
    constexpr uint32_t SEM_OWN_STATS_BASE = get_compile_time_arg_val(18);
    constexpr uint32_t SEG_STRIDE = get_compile_time_arg_val(20);
    constexpr uint32_t GB = get_compile_time_arg_val(22);
    constexpr uint32_t PW = get_compile_time_arg_val(23);
    constexpr uint32_t INV_FACE = T_I / 4;
    constexpr uint32_t INV_HALF = 16 * (T_I / 1024);
    static_assert(PW == NW, "dedicated owners: one pass only");

    size_t a = 0;
    const uint32_t rows = get_arg_val<uint32_t>(a++);
    const uint32_t my_pos = get_arg_val<uint32_t>(a++);
    const uint32_t landing = get_arg_val<uint32_t>(a++);
    const uint32_t group = get_arg_val<uint32_t>(a++);
    const uint32_t n_in_fwd = get_arg_val<uint32_t>(a++);
    const uint32_t n_in_bwd = get_arg_val<uint32_t>(a++);
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
    auto n_own = [&](uint32_t w, uint32_t k) -> uint32_t {
        const uint32_t n = n_wave(w);
        const uint32_t first = k * Q;
        return first >= n ? 0 : ((n - first) < Q ? n - first : Q);
    };

    // slots with rows, (w, k) order
    uint8_t sw[MAX_SLOTS], sk[MAX_SLOTS];
    uint32_t ns = 0;
    for (uint32_t w = 0; w < NW; ++w) {
        for (uint32_t k = 0; k < OWNERS; ++k) {
            if (n_own(w, k) > 0) {
                sw[ns] = w;
                sk[ns] = k;
                ++ns;
            }
        }
    }

    Noc noc;
    {
        DRMS_ZONE("ow_prologue");
        // chip_sum's reduce takes the scaler CB (1.0)
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            cb_reduce_scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW>();
        // records land in row 0 of faces 0 / 1 of the cb_stats pages: the rest stays zero for the whole call
        if constexpr (!DRMS_OWNER_PUBLISH_ONLY) {
            CircularBuffer stats_cb(cb_stats);
            noc.async_write_zeros(stats_cb, G * Q * STATS_TILE_BYTES, {.offset_bytes = 0});
            noc.write_zeros_l1_barrier();
        }
    }
    const uint32_t scratch = get_write_ptr(cb_record_scratch);
    const uint32_t inv_base = get_write_ptr(cb_inv_rms);  // the members' cb_inv_rms base (same CB descriptor)
    volatile tt_l1_ptr uint32_t* gs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GATHER));

    auto link_of = [&](uint32_t k) { return (group * OWNERS + k) % L; };

    auto gather_done = [&](uint32_t s) -> bool { return ((*gs >> (GB * s)) & ((1u << GB) - 1)) >= G_W; };
    auto landed_ok = [&](uint32_t w, uint32_t l) -> bool {
        if constexpr (DRMS_ABLATE_EXCHANGE) {
            return true;
        }
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

    // publish (w, k): the port counts every owner slot of every wave (own_stats[w] >= publishers of the link), so a
    // slot without rows (a shorter row-group's last wave) publishes zero records, as the baseline owner does
    auto publish_wk = [&](uint32_t w, uint32_t k) {
        DRMS_ZONE("ow_publish");
        const uint32_t no = n_own(w, k);
        const uint32_t kp = group * OWNERS + k;
        const uint32_t l = kp % L;
        const uint32_t kl = kp / L;
        volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        const uint32_t src = no > 0 ? get_read_ptr(cb_chip_partial) : 0;
        {
            DRMS_ZONE("ow_extract");
            // all 32 column-0 loads of a row first, then the stores (no load-use stall per element; Perf 1's
            // unpack_inv pattern)
            for (uint32_t r = 0; r < no; ++r) {
                const uint32_t tile = src + r * T_I;
                uint32_t v[32];
                if constexpr (INTERM_FP32) {
                    volatile tt_l1_ptr uint32_t* t32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile);
#pragma GCC unroll 32
                    for (uint32_t i = 0; i < 32; ++i) {
                        v[i] = t32[col0_index(i)];
                    }
                } else {
                    volatile tt_l1_ptr uint16_t* t16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile);
#pragma GCC unroll 32
                    for (uint32_t i = 0; i < 32; ++i) {
                        v[i] = static_cast<uint32_t>(t16[col0_index(i)]) << 16;
                    }
                }
#pragma GCC unroll 32
                for (uint32_t i = 0; i < 32; ++i) {
                    rec[r * 32 + i] = v[i];
                }
            }
        }
        if (no > 0) {
            cb_pop_front(cb_chip_partial, no);
        }
        for (uint32_t r = no; r < Q; ++r) {
            for (uint32_t i = 0; i < 32; ++i) {
                rec[r * 32 + i] = 0;
            }
        }
        const uint32_t off = landing + (w * G + my_pos) * SEG_STRIDE + kl * Q * RECORD_BYTES;
        if (sends_own_fwd) {
            noc_async_write(scratch, port_noc(l, FWD, off), Q * RECORD_BYTES);
        }
        if (sends_own_bwd) {
            noc_async_write(scratch, port_noc(l, BWD, off), Q * RECORD_BYTES);
        }
        noc_async_write_barrier();
        const uint32_t own_stats = get_semaphore(SEM_OWN_STATS_BASE + w);
        if (sends_own_fwd) {
            noc_semaphore_inc(port_noc(l, FWD, own_stats), 1);
        }
        if (sends_own_bwd) {
            noc_semaphore_inc(port_noc(l, BWD, own_stats), 1);
        }
        if constexpr (DRMS_OWNER_PUBLISH_ONLY) {
            // the member that collects (w, k) (the baseline's rotating owner) reads this chip's record from the port
            // too: its landed notifies cover only the remote entries, so tell it the own record is in the port
            if (no > 0) {
                const uint32_t m = (w * OWNERS + k) % G_W;
                noc_semaphore_inc(get_noc_addr(member_x(m), member_y(m), get_semaphore(SEM_INV)), 1);
            }
        }
    };

    auto publish = [&](uint32_t s) { publish_wk(sw[s], sk[s]); };
    // slots without rows: zero records at once (no data dependency)
    for (uint32_t w = 0; w < NW; ++w) {
        for (uint32_t k = 0; k < OWNERS; ++k) {
            if (n_own(w, k) == 0) {
                publish_wk(w, k);
            }
        }
    }

    auto collect = [&](uint32_t s) {
        DRMS_ZONE("ow_collect");
        const uint32_t w = sw[s], k = sk[s];
        const uint32_t no = n_own(w, k);
        const uint32_t kp = group * OWNERS + k;
        const uint32_t l = kp % L;
        const uint32_t kl = kp / L;
        cb_reserve_back(cb_stats, G * Q);
        const uint32_t st = get_write_ptr(cb_stats);
        for (uint32_t pos = 0; pos < G; ++pos) {
            uint32_t d = get_arg_val<uint32_t>(srcdir_idx + pos);
            if (d == OWN) {
                d = sends_own_fwd ? FWD : BWD;
            }
            const uint32_t off = landing + (w * G + pos) * SEG_STRIDE + kl * Q * RECORD_BYTES;
            for (uint32_t r = 0; r < no; ++r) {
                const uint32_t page = st + (pos * Q + r) * STATS_TILE_BYTES;
                noc_async_read(port_noc(l, d, off + r * RECORD_BYTES), page, RECORD_BYTES / 2);
                noc_async_read(
                    port_noc(l, d, off + r * RECORD_BYTES + RECORD_BYTES / 2),
                    page + STATS_FACE_BYTES,
                    RECORD_BYTES / 2);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_stats, G * Q);
    };

    auto allgather = [&](uint32_t s) {
        DRMS_ZONE("ow_allgather");
        const uint32_t w = sw[s], k = sk[s];
        const uint32_t no = n_own(w, k);
        const uint32_t src = get_read_ptr(cb_rms_root);
        const uint32_t dst = inv_base + ((w % PW) * Q_W + k * Q) * T_I;
        for (uint32_t r = 0; r < no; ++r) {
            volatile tt_l1_ptr uint32_t* pg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + r * T_I);
            uint32_t v[INV_HALF / 4];
#pragma GCC unroll 16
            for (uint32_t i = 0; i < INV_HALF / 4; ++i) {
                v[i] = pg[INV_FACE / 4 + i];
            }
#pragma GCC unroll 16
            for (uint32_t i = 0; i < INV_HALF / 4; ++i) {
                pg[INV_HALF / 4 + i] = v[i];
            }
        }
        for (uint32_t m = 0; m < G_W; ++m) {
            noc_async_write_one_packet_set_state<true>(get_noc_addr(member_x(m), member_y(m), dst), 2 * INV_HALF);
            for (uint32_t r = 0; r < no; ++r) {
                noc_async_write_one_packet_with_state<true>(src + r * T_I, dst + r * T_I + INV_FACE);
            }
        }
        noc_async_posted_writes_flushed();
        cb_pop_front(cb_rms_root, no);
    };

    // in-order queues: sa <= sc <= sp <= sg <= ns
    uint32_t sg = 0, sp = 0, sc = 0, sa = 0;
    // 0 allgather, 1 collect, 2 publish, 3 gather push, 4 none (downstream first: the chain's tail is the critical one)
    auto ready = [&]() -> uint32_t {
        invalidate_l1_cache();
        if (!DRMS_OWNER_PUBLISH_ONLY && sa < sc && cb_pages_available_at_front(cb_rms_root, n_own(sw[sa], sk[sa]))) {
            return 0;
        }
        if (!DRMS_OWNER_PUBLISH_ONLY && sc < sp && landed_ok(sw[sc], link_of(sk[sc])) &&
            cb_pages_reservable_at_back(cb_stats, G * Q)) {
            return 1;
        }
        if (sp < sg && cb_pages_available_at_front(cb_chip_partial, n_own(sw[sp], sk[sp]))) {
            return 2;
        }
        if (sg < ns && gather_done(sg)) {
            return 3;
        }
        return 4;
    };
    while (DRMS_OWNER_PUBLISH_ONLY ? sp < ns : sa < ns) {
        uint32_t r = ready();
        if (r == 4) {
            DRMS_ZONE("ow_wait");
            do {
                r = ready();
            } while (r == 4);
        }
        if (r == 0) {
            allgather(sa++);
        } else if (r == 1) {
            collect(sc++);
        } else if (r == 2) {
            publish(sp++);
        } else {
            DRMS_ZONE("ow_gather_push");
            cb_reserve_back(cb_gather, G_W * Q);
            cb_push_back(cb_gather, G_W * Q);
            ++sg;
        }
    }
    noc_async_write_barrier();
    noc_async_posted_writes_flushed();
    noc_async_atomic_barrier();
}
