// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// nomic_sdpa — reader (RISCV_1, NOC0).
//
// Split reads (Perf 1, perf_experiments/dual_noc_reads): with CT arg split_reads (the default) the Q and V streams are
// read by the WRITER RISC (BRISC) on NOC1, so the read flood uses both NoCs' links (8x512 L1: DM floor ~69 -> ~44 us).
// This reader then neither reserves nor pushes cb_query / cb_value_aug (single-producer invariant) and keeps K, the K
// multicast and the V multicast SENDS (NOC0, first-core sender, unchanged routing); on a V sender core it raises local
// flag kv_sem_base + 2 * kv_depth + slot once the slot's V is in L1, and the writer pushes the slot on it.
// split_reads = 0 (host carve-out: a 1-deep cb_query, i.e. DRAM, with S_t <= 8 — measured slower split) keeps every
// read here, the dataflow described below.
//
// Split startup (Perf 2, perf_experiments/startup_k_first; under split_reads): the first segment's K and V — the t = 0
// burst every core issues at once — are each read half / half on both NoCs into the same ring slot. One RISC's 64 KiB
// of K on NOC0 landed at ~16-18 us at 8x512 (110 cores reading at t = 0) even with every other read deferred: the time
// is NOC0 congestion, superlinear in the bytes one NoC carries. Split, each NoC carries K/2 then V/2 a core and the
// first K lands at ~7-8 us (8x512 L1 whole op ~71 -> ~65 us at the time).
//   K   this RISC reads key rows [0, k_split_rows) on NOC0, the writer the rest on NOC1 and raises same-core flag
//       k_half once they are in L1; this RISC — the sole cb_key_t producer — then pushes (READ) or multicasts (SEND).
//       Host carve-out (Plan.k_split_rows = N: no K split, V still split): a small t = 0 K burst, measured slower. Both
//       RISCs write the slot before its reserve: the same argument as the pre-send (segment 0's slot holds no earlier
//       data).
//   V   this RISC reads value rows [0, N/2), the writer [N/2, N) plus the ones column. READ: this RISC raises the
//       slot's V flag (the sender hand-off flag below) and the writer pushes once both halves are in. SEND: the writer
//       raises v_half, this RISC multicasts the whole slot, then raises the V flag as before.
// The flags are raw volatile L1 semaphores, like the V hand-off: kernel_lib has no same-core RISC-to-RISC signal.
//
// Throttled startup (Perf 2, perf_experiments/startup_first_qk; CT arg read_chunk > 0, host gate: a t = 0 K burst of
// more than 1536 tiles): every segment-0 page read of this RISC (K, V, the Q lead rows) is retired in chunks of
// read_chunk pages — a read barrier after each chunk — so at most read_chunk of its reads are in flight. With ~200
// RISCs each issuing 16-32 reads at t = 0 the unthrottled NoC collapses: 8x512 L1 whole op 55.1 .. 67.1 us run to run
// (first K landed at 5-11 us, first V at 9-22 us); chunks of 6: 53.1 .. 57.2 us, median 60.9 -> 53.6 (first K ~4.4 us,
// first V ~8 us). Alternatives measured at 8x512 (median of 11): a sliding window of 8 reads in flight (raw NIU read
// counters) 55.1, two trid-tagged batches of 3 / 4 (kernel_lib set_read_trid) 54.5 / 54.6, chunks of 6 54.3: the plain
// barrier is both the simplest and the fastest. Under the throttle the reader takes 11/16 of segment 0's key rows (host
// k_split_rows): the writer reads the Q lead rows ahead of its part (8x512 54.0 -> 53.5 us).
//
// Per segment (one (b, h) pair on this core):
//   load_kv_block   K of the pair into cb_key_t in K^T tile order (slot dt*N + st): all N*HD page reads, one
//                   barrier, one push. V into cb_value_aug at slot st*PV + dt (dt < HD), one barrier, one push; slot
//                   st*PV + HD is the ones column (an all-ones tile, so P V carries the row sum l in every column for
//                   compute's normalize in DEST), written once per depth slot — under the V reads when this core
//                   reads its own V.
// Per block:
//   load_query_block  the block's rows (pages 2g + dt, contiguous) into cb_query, in two pushes inside one reserve:
//                     the first q_lead_rows rows, then the rest; the NOMINAL q_block_rows*HD pages in total (ring
//                     discipline: every block advances the ring by its capacity).
// Order inside a block: the lead Q rows, K, V (first block of a segment), then the rest of Q. Compute consumes Q a row
// at a time and the row-group schedule needs V from its second step on, so the startup burst every core issues at
// t = 0 is lead Q + K + V rather than the whole Q block + K + V (Refinement 4: the first V landed at ~46 us at 8x512
// behind ~56 KiB a core of Q that compute reads over the next ~70 us).
//
// K/V pair multicast (Refinement 2): segment k < kv_depth of a core has a per-core role (RT args, host
// kv_multicast_topology): READ (own page reads, as above), SEND (own page reads, then multicast the slot to the pair's
// group) or RECEIVE (wait for the group sender's data-ready flag). Both rings start p0 % kv_depth phantom (data-less)
// slots in, so pair p always lands in ring slot p % kv_depth — the identical address on every group member — and a
// segment k < kv_depth lands in a slot no earlier segment of the core used, so no readiness handshake is needed.
// Transport: kernel_lib SenderPipe / ReceiverPipe (mcast_pipe.hpp) with no pre-handshake and one Flag semaphore per
// (slot, K | V). The host Mcast / McastArgs channel cannot express this topology: its groups are equal-size disjoint
// partitions, while a pair group here is an irregular core run (size set by the row table) and a core belongs to up to
// kv_depth groups, so the pipes are built straight from per-core RT args (the rectangles in the wire's field order).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/pipes/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "nomic_sdpa_schedule.hpp"

// Permanent per-stage device zones (Perf 1): reader_q_reserve (back-pressure from compute on cb_query), reader_q_lead /
// reader_q_rest (Q issue + barrier), reader_presend_k / reader_presend_v (the sender's read + multicast), reader_k /
// reader_v (own read, or the multicast receive wait), reader_v_ones (the ones column).

namespace {

using namespace dataflow_kernel_lib;

constexpr uint32_t kRoleRead = 0, kRoleSend = 1, kRoleReceive = 2;  // host ROLE_* (program descriptor)

// Call fn(std::integral_constant<uint32_t, slot>) for a runtime ring slot < Depth: the pipes bind their semaphore id
// at compile time.
template <uint32_t Depth, typename Fn>
FORCE_INLINE void with_slot(uint32_t slot, Fn&& fn) {
    if constexpr (Depth > 0) {
        if (slot == Depth - 1) {
            fn(std::integral_constant<uint32_t, Depth - 1>{});
        } else {
            with_slot<Depth - 1>(slot, fn);
        }
    }
}

template <uint32_t SemId, uint32_t MaxRects>
using KvSender = SenderPipe<
    noc_index,
    SemId,
    false,
    UNUSED_SEM_ID,
    DataReadySignal::Flag,
    false,
    SenderMcastMode::Unknown,
    MaxRects>;
template <uint32_t SemId>
using KvReceiver = ReceiverPipe<SemId, false>;

}  // namespace

#ifdef ABLATE_READS
#define READ_PAGE(...) ((void)0)  // ABLATION ONLY: keep the CB protocol, drop the NoC reads
#else
#define READ_PAGE(...) noc_async_read_page(__VA_ARGS__)
#endif

void kernel_main() {
    constexpr uint32_t cb_query = get_named_compile_time_arg_val("cb_query");
    constexpr uint32_t cb_key_t = get_named_compile_time_arg_val("cb_key_t");
    constexpr uint32_t cb_value_aug = get_named_compile_time_arg_val("cb_value_aug");
    constexpr uint32_t Q = get_named_compile_time_arg_val("q_block_rows");
    constexpr uint32_t N = get_named_compile_time_arg_val("key_block_tiles");
    constexpr uint32_t HD = get_named_compile_time_arg_val("head_dim_tiles");
    constexpr uint32_t PV = get_named_compile_time_arg_val("pv_out_tiles");
    constexpr uint32_t seq_tiles = get_named_compile_time_arg_val("seq_tiles");
    constexpr uint32_t kv_depth = get_named_compile_time_arg_val("kv_depth");
    constexpr uint32_t grid_x = get_named_compile_time_arg_val("grid_x");
    constexpr uint32_t kv_sem_base = get_named_compile_time_arg_val("kv_sem_base");
    constexpr uint32_t max_rects = get_named_compile_time_arg_val("max_mcast_rects");
    // Q rows a block reads ahead of the segment's K / V (host knob Q_LEAD_ROWS; >= Q: the whole block first).
    constexpr uint32_t q_lead_rows = get_named_compile_time_arg_val("q_lead_rows");
    constexpr bool kv_bank_stagger = get_named_compile_time_arg_val("kv_bank_stagger") == 1;
    constexpr bool split_reads = get_named_compile_time_arg_val("split_reads") == 1;
    constexpr bool own_q = !split_reads;  // cb_query producer: this RISC, or the writer under split_reads
    constexpr bool own_v = !split_reads;  // cb_value_aug producer (own reads, receives, ones column)
    // Split startup (file header): the key rows of segment 0 this RISC reads (N: all of them, no K split), and the
    // same-core flags from the writer (its K half / its V half of a SEND slot are in L1).
    constexpr uint32_t k_split_rows = get_named_compile_time_arg_val("k_split_rows");
    constexpr bool split_k = split_reads && k_split_rows < N;
    constexpr bool split_v = split_reads && N >= 2;  // (one key row: nothing to split)
    constexpr uint32_t sem_k_half = kv_sem_base + 3 * kv_depth;
    constexpr uint32_t sem_v_half = kv_sem_base + 3 * kv_depth + 1;
    static_assert(q_lead_rows > 0, "compute needs at least one Q row before anything else");
    static_assert(PV == HD + 1, "cb_value_aug rows hold HD value tiles and one ones column");

    constexpr auto q_args = TensorAccessorArgs<0>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

    constexpr uint32_t kTableArg = get_named_compile_time_arg_val("table_arg");  // after the io addresses
    const uint32_t q_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t core = nomic_sdpa::core_index(grid_x);
    const uint32_t row_begin = get_common_arg_val<uint32_t>(kTableArg + core);
    const uint32_t row_end = get_common_arg_val<uint32_t>(kTableArg + core + 1);

    constexpr uint32_t tile_bytes = get_tile_size(cb_query);
    const auto q_acc = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_acc = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v_acc = TensorAccessor(v_args, v_addr, tile_bytes);

    Noc noc;

    // Throttled startup (file header): a read barrier after every read_chunk segment-0 page reads (0: unthrottled).
    constexpr uint32_t read_chunk = get_named_compile_time_arg_val("read_chunk");
    uint32_t chunk_reads = 0;
    auto throttle = [&](bool segment0) {
        if constexpr (read_chunk > 0) {
            if (segment0 && ++chunk_reads == read_chunk) {
                chunk_reads = 0;
                MaybeDeviceZoneScope("reader_read_chunk");
                noc_async_read_barrier();
            }
        }
    };

    // ---- K/V multicast roles (per-core RT args: kv_depth roles, then kv_depth sender blocks) ----
    constexpr uint32_t sender_words = 1 + max_rects * mcast_wire::RECT_WORDS;
    uint32_t roles[kv_depth];
    for (uint32_t k = 0; k < kv_depth; ++k) {
        roles[k] = get_arg_val<uint32_t>(k);
    }
    auto sender_args = [&](uint32_t k) {
        const uint32_t base = kv_depth + k * sender_words;
        SenderRuntimeArgumentsFor<max_rects> args{};
        args.num_rectangles = get_arg_val<uint32_t>(base);
        for (uint32_t i = 0; i < args.num_rectangles; ++i) {
            const uint32_t w = base + 1 + i * mcast_wire::RECT_WORDS;
            const uint32_t lo_x = get_arg_val<uint32_t>(w), lo_y = get_arg_val<uint32_t>(w + 1);
            const uint32_t hi_x = get_arg_val<uint32_t>(w + 2), hi_y = get_arg_val<uint32_t>(w + 3);
            auto& rect = args.rectangles[i];
            // Routing order: NOC0 multicasts from the low corner, NOC1 from the high one.
            rect.bounds = noc_index == 0 ? NocBounds{lo_x, lo_y, hi_x, hi_y} : NocBounds{hi_x, hi_y, lo_x, lo_y};
            rect.remote_count = get_arg_val<uint32_t>(w + 4);
            rect.loopback_count = get_arg_val<uint32_t>(w + 5);
            rect.sender_mcast_mode = static_cast<SenderMcastMode>(get_arg_val<uint32_t>(w + 6));
        }
        return args;
    };
    const uint32_t no_sender_coords[mcast_wire::SENDER_COORD_WORDS] = {};  // no handshake: never addressed

    // Phantom slots: pair p lands in ring slot p % kv_depth on every core (see the file header). Compute pops them.
    const uint32_t key_ring_base = get_write_ptr(cb_key_t);
    const uint32_t value_ring_base = get_write_ptr(cb_value_aug);
    constexpr uint32_t key_slot_bytes = N * HD * tile_bytes;
    constexpr uint32_t value_slot_bytes = N * PV * tile_bytes;
    const uint32_t first_pair = row_begin / seq_tiles;
    const uint32_t num_segments = (row_end - 1) / seq_tiles - first_pair + 1;
    const uint32_t phantoms = first_pair % kv_depth;
    for (uint32_t i = 0; i < phantoms; ++i) {
        cb_reserve_back(cb_key_t, N * HD);
        cb_push_back(cb_key_t, N * HD);
        if constexpr (own_v) {
            cb_reserve_back(cb_value_aug, N * PV);
            cb_push_back(cb_value_aug, N * PV);
        }
    }

    // load_kv_block's page reads: K in K^T tile order for matmul in1 [k = dt][n = st]; V rows at slot st*PV + dt.
    // L9 bank stagger (host knob KV_BANK_STAGGER, catalog bank_stagger): start each core's K / V key-row walk at row
    // core % N and wrap, so the cores reading one pair at the same time start on different L1 / DRAM banks.
    // A split read walks only its own rows [0, rows), rotated inside that range.
    const uint32_t kv_row_rotation = kv_bank_stagger ? core % N : 0;
    auto wait_flag = [](uint32_t sem) {  // a same-core flag from the writer (file header), consumed
        volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem));
        while (*flag == 0) {
            invalidate_l1_cache();
        }
        *flag = 0;
    };
    // first_segment: segment 0 under split_k — this RISC reads key rows [0, k_split_rows), then waits for the writer's.
    auto read_key = [&](uint32_t pair, uint32_t k_base, bool first_segment) {
        const uint32_t pair_first_page = pair * seq_tiles * HD;
        const uint32_t rows = split_k && first_segment ? k_split_rows : N;
        const uint32_t rotation = kv_row_rotation % rows;
        {
            MaybeDeviceZoneScope("reader_kv_issue");
            for (uint32_t i = 0; i < rows; ++i) {
                const uint32_t st = i + rotation < rows ? i + rotation : i + rotation - rows;
                for (uint32_t dt = 0; dt < HD; ++dt) {
                    READ_PAGE(pair_first_page + st * HD + dt, k_acc, k_base + (dt * N + st) * tile_bytes);
                    throttle(first_segment);
                }
            }
        }
        {
            MaybeDeviceZoneScope("reader_kv_barrier");
            noc_async_read_barrier();
        }
        if (rows < N) {
            MaybeDeviceZoneScope("reader_k_wait_half");
            wait_flag(sem_k_half);
        }
    };
    // value rows [0, rows) (N: all of them; N / 2: this RISC's half of a split segment-0 V) of segment `segment`
    auto read_value = [&](uint32_t pair, uint32_t v_base, uint32_t rows, uint32_t segment, auto&& under_reads) {
        const uint32_t pair_first_page = pair * seq_tiles * HD;
        const uint32_t rotation = kv_row_rotation % rows;
        {
            MaybeDeviceZoneScope("reader_kv_issue");
            for (uint32_t i = 0; i < rows; ++i) {
                const uint32_t st = i + rotation < rows ? i + rotation : i + rotation - rows;
                for (uint32_t dt = 0; dt < HD; ++dt) {
                    READ_PAGE(pair_first_page + st * HD + dt, v_acc, v_base + (st * PV + dt) * tile_bytes);
                    throttle(segment == 0);
                }
            }
        }
        under_reads();  // work that overlaps the reads in flight (the ones fill of a READ segment)
        MaybeDeviceZoneScope("reader_kv_barrier");
        noc_async_read_barrier();
    };

    // ---- pre-send: every pair this core multicasts is read into its fresh slot and sent ahead of the schedule (the
    // slot is pushed later, in schedule order, without a re-read). Writing a slot before its reserve is safe: segment
    // k < kv_depth's slot holds no earlier segment's data (at most a data-less phantom compute never reads). Order by
    // when the group needs it: every K at t = 0 (qk_block needs it first), every V at the first segment's V half,
    // after this core's own first Q (pv_block needs V last) — so the chip-wide t = 0 burst is K + Q, not K + V + Q.
    auto for_each_send = [&](auto&& fn) {
        for (uint32_t k = 0; k < kv_depth && k < num_segments; ++k) {
            if (roles[k] == kRoleSend) {
                const uint32_t pair = first_pair + k;
                fn(k, pair, pair % kv_depth);
            }
        }
    };
    auto presend_keys = [&]() {
        for_each_send([&](uint32_t k, uint32_t pair, uint32_t slot) {
            const uint32_t k_base = key_ring_base + slot * key_slot_bytes;
            read_key(pair, k_base, k == 0);
            const auto args = sender_args(k);
            with_slot<kv_depth>(slot, [&](auto s) {
                KvSender<kv_sem_base + 2 * decltype(s)::value, max_rects>(noc, args).send(
                    k_base, k_base, key_slot_bytes);
            });
        });
    };
    auto presend_values = [&]() {
        for_each_send([&](uint32_t k, uint32_t pair, uint32_t slot) {
            const uint32_t v_base = value_ring_base + slot * value_slot_bytes;
            if (split_v && k == 0) {  // segment 0: rows [0, N / 2) here, the writer's [N / 2, N) on its flag
                read_value(pair, v_base, N / 2, k, [] {});
                MaybeDeviceZoneScope("reader_v_wait_half");
                wait_flag(sem_v_half);
            } else {
                read_value(pair, v_base, N, k, [] {});
            }
            const auto args = sender_args(k);
            with_slot<kv_depth>(slot, [&](auto s) {
                KvSender<kv_sem_base + 2 * decltype(s)::value + 1, max_rects> sender(noc, args);
                for (uint32_t st = 0; st < N;
                     ++st) {  // the HD value tiles of each row, contiguous; never the ones tile
                    const uint32_t row = v_base + st * PV * tile_bytes;
                    sender.send_payload(row, row, HD * tile_bytes);
                }
                sender.send_signal();  // publishes the payloads (same NoC, issued later); waits the source read
            });
            if constexpr (!own_v) {  // the slot's V is in L1 (read_value barriered): hand it to the writer
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(kv_sem_base + 2 * kv_depth + slot)) = 1;
            }
        });
    };
    {
        MaybeDeviceZoneScope("reader_presend_k");
        presend_keys();
    }

    // The ones column of one cb_value_aug depth slot: tile (0, HD) filled with bf16 1.0 (512 word stores), then
    // copied L1 -> L1 into every other row's slot. All 32 columns: [O | l] = P [V | 1] then carries l in every column,
    // so compute divides O by l lane-wise in DEST (no broadcast).
    auto write_ones_column = [&](uint32_t slot_base) {
        const uint32_t ones_tile = slot_base + HD * tile_bytes;
        volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ones_tile);
#pragma GCC unroll 16
        for (uint32_t i = 0; i < tile_bytes / 4; ++i) {
            words[i] = 0x3F803F80;  // bf16 1.0 in every element: P V carries l in every column (normalize in DEST)
        }
        if constexpr (N > 1) {
            dataflow_kernel_lib::set_read_state<tile_bytes>(noc, ones_tile);
            for (uint32_t st = 1; st < N; ++st) {
                dataflow_kernel_lib::read_with_state<tile_bytes>(noc, ones_tile + st * PV * tile_bytes, ones_tile);
            }
            // Retire the copies here: a RECEIVE / SEND segment pushes the slot with no V read barrier after this.
            noc_async_read_barrier();
        }
    };

    nomic_sdpa::for_each_block<seq_tiles, Q>(row_begin, row_end, [&](const nomic_sdpa::Block& blk) {
        const uint32_t role = blk.segment < kv_depth ? roles[blk.segment] : kRoleRead;
        const uint32_t slot = blk.pair % kv_depth;
        // load_query_block: rows [row0, row0 + rows) are pages [2*row0, 2*(row0 + rows)). The first `lead` rows go
        // ahead of the segment's K / V, the rest after them (compute consumes Q a row at a time, so it starts on the
        // lead rows; the rest is not on the startup's critical path). Pushes are cumulative within the one reserve.
        if constexpr (own_q) {
            MaybeDeviceZoneScope("reader_q_reserve");
            cb_reserve_back(cb_query, Q * HD);
        }
        const uint32_t q_base = get_write_ptr(cb_query);
        const uint32_t lead = blk.rows < q_lead_rows ? blk.rows : q_lead_rows;
        const bool startup = blk.segment == 0 && blk.first_of_segment;  // the throttled first block
        auto load_query_rows = [&](uint32_t first_row, uint32_t end_row) {
            for (uint32_t i = first_row * HD; i < end_row * HD; ++i) {
                READ_PAGE(blk.row0 * HD + i, q_acc, q_base + i * tile_bytes);
                throttle(startup && first_row == 0);  // the lead rows only
            }
            noc_async_read_barrier();
        };
        if constexpr (own_q) {
            {
                MaybeDeviceZoneScope("reader_q_lead");
                load_query_rows(0, lead);
            }
            cb_push_back(cb_query, lead * HD);
        }

        if (blk.first_of_segment) {
            // load_kv_block (K half). SEND: already read (and multicast) by the pre-send.
            cb_reserve_back(cb_key_t, N * HD);
            const uint32_t k_base = get_write_ptr(cb_key_t);
            ASSERT(k_base == key_ring_base + slot * key_slot_bytes);
            MaybeDeviceZoneScope("reader_k");
            if (role == kRoleReceive) {
                with_slot<kv_depth>(slot, [&](auto s) {
                    KvReceiver<kv_sem_base + 2 * decltype(s)::value>(noc, no_sender_coords).receive();
                });
            } else if (role == kRoleRead) {
                read_key(blk.pair, k_base, blk.segment == 0);
            }
            cb_push_back(cb_key_t, N * HD);
        }

        if (blk.first_of_segment) {
            // load_kv_block (V half): [V | 1] rows, slot st*PV + dt.
            if (blk.segment == 0) {
                MaybeDeviceZoneScope("reader_presend_v");
                presend_values();
            }
            if (split_v && blk.segment == 0 && role == kRoleRead) {
                // value rows [0, N / 2) of the fresh slot (no reserve: the writer owns cb_value_aug and pushes the slot
                // once its half is in too), then the slot's V flag
                MaybeDeviceZoneScope("reader_v_half");
                read_value(blk.pair, value_ring_base + slot * value_slot_bytes, N / 2, blk.segment, [] {});
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(kv_sem_base + 2 * kv_depth + slot)) = 1;
            }
        }
        if (own_v && blk.first_of_segment) {
            cb_reserve_back(cb_value_aug, N * PV);
            const uint32_t v_base = get_write_ptr(cb_value_aug);
            ASSERT(v_base == value_ring_base + slot * value_slot_bytes);
            // First real use of this ring slot (phantoms carry no data); pushes stay nominal, so it persists. The
            // multicast never touches the ones tiles: it moves only the HD value tiles of each row. A READ segment
            // fills them while its own V reads are in flight.
            const bool ones_now = blk.segment < kv_depth;
            if (ones_now && role != kRoleRead) {
                MaybeDeviceZoneScope("reader_v_ones");
                write_ones_column(v_base);
            }
            MaybeDeviceZoneScope("reader_v");
            if (role == kRoleReceive) {
                with_slot<kv_depth>(slot, [&](auto s) {
                    KvReceiver<kv_sem_base + 2 * decltype(s)::value + 1>(noc, no_sender_coords).receive();
                });
            } else if (role == kRoleRead) {
                read_value(blk.pair, v_base, N, blk.segment, [&] {
                    if (ones_now) {
                        write_ones_column(v_base);
                    }
                });
            }
            cb_push_back(cb_value_aug, N * PV);
        }

        // load_query_block (the rest), then close the ring with the NOMINAL q_block_rows*HD pages (pads carry no data).
        if constexpr (own_q) {
            if (lead < blk.rows) {
                MaybeDeviceZoneScope("reader_q_rest");
                load_query_rows(lead, blk.rows);
            }
            cb_push_back(cb_query, (Q - lead) * HD);
        }
    });
    noc_async_write_barrier();
}
