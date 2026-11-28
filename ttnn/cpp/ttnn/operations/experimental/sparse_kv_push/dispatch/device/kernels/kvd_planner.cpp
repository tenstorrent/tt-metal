// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// sparse_kv_dispatch planner (BRISC, planner cores): once every chip's slice has landed in this chip's inbox, turns the
// slices of a contiguous range of this chip's blocks into each row's destination mask, copies the rows this chip itself
// needs into `output`, and writes every other needed row's first hops into the own lists of the ports that send them.
// Then it publishes its list lengths and tells every port core.
//
// A slice (inbox slot i) is chip i's plan-table slot for this owner: plan_table header, then 5 bitmap words per local
// block (160 rows / 32). Row g of local block lb, word w, bit k: g = (me + lb * N) * stripe + w * 32 + k.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/common/kernels/sparse_push_constants.hpp"
#include "api/debug/kernel_profiler.hpp"
#include "api/debug/dprint.h"
#include "kvd_common.hpp"

#include <cstdint>

using namespace kvd;

namespace {
constexpr uint8_t kXRing[] = {KVD_XRING};
constexpr uint8_t kYRing[] = {KVD_YRING};
}  // namespace

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t R = get_compile_time_arg_val(1);
    constexpr uint32_t C = get_compile_time_arg_val(2);
    constexpr bool WX = get_compile_time_arg_val(3) != 0;
    constexpr bool WY = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t lanes = get_compile_time_arg_val(5);  // lanes (links x lanes per link)
    constexpr uint32_t stripe = get_compile_time_arg_val(6);
    constexpr uint32_t owner_slot_rows = get_compile_time_arg_val(7);  // output rows per owner (owner-major pages)
    constexpr uint32_t inbox_slot_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t planners = get_compile_time_arg_val(10);
    constexpr uint32_t pages_own = get_compile_time_arg_val(11);
    constexpr uint32_t pages_per_slot = get_compile_time_arg_val(12);  // cache pages per slot
    constexpr bool has_batch = get_compile_time_arg_val(13) != 0;
    constexpr uint32_t num_slots = get_compile_time_arg_val(14);
    constexpr uint32_t owner_blocks_max = get_compile_time_arg_val(15);
    constexpr uint32_t cb_inbox = get_compile_time_arg_val(16);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(17);
    constexpr uint32_t cb_row = get_compile_time_arg_val(18);
    constexpr auto input_args = TensorAccessorArgs<19>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto inbox_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto scratch_args = TensorAccessorArgs<inbox_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<scratch_args.next_compile_time_args_offset()>();
    constexpr auto batch_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr uint32_t words_per_block = stripe / 32;
    constexpr uint32_t bitmap_off = ::sparse_push::plan_table::kBitmapOff;

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t output_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t inbox_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t scratch_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t batch_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t num_layers = get_common_arg_val<uint32_t>(5);
    const uint32_t layer_idx = get_common_arg_val<uint32_t>(6);
    const uint32_t host_slot = get_common_arg_val<uint32_t>(7);

    uint32_t a = 0;
    const uint32_t planner = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);
    const uint32_t my_r = me / C, my_c = me % C;
    auto* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(a++)));
    const uint32_t own_ready_sem = get_semaphore(get_arg_val<uint32_t>(a++));  // on the port cores
    const uint32_t ports_present = get_arg_val<uint32_t>(a++);                 // bit p: port core p exists
    uint32_t port_x[kMaxPorts], port_y[kMaxPorts];
    for (uint32_t p = 0; p < kMaxPorts; ++p) {
        port_x[p] = get_arg_val<uint32_t>(a++);
        port_y[p] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t count_base = get_arg_val<uint32_t>(a++);  // scratch page of port 0's list counts
    const uint32_t is_sync = get_arg_val<uint32_t>(a++);     // planner 0: owns the inbox word
    const uint32_t go_sem_addr = reinterpret_cast<uint32_t>(go);
    const uint32_t planner_xy_arg = a;  // planners x (noc x, noc y), read when needed (no stack copy)
    a += 2 * planners;
    const uint32_t table_addr = get_common_arg_val<uint32_t>(12);
    const uint32_t inbox_sem_addr = get_common_arg_val<uint32_t>(11);

    const auto input = TensorAccessor(input_args, input_addr, row_bytes);
    const auto output = TensorAccessor(output_args, output_addr, row_bytes);
    const auto inbox = TensorAccessor(inbox_args, inbox_addr, inbox_slot_bytes);
    const auto scratch = TensorAccessor(scratch_args, scratch_addr, kScratchPageBytes);

    // The cache slot: the host's, or (trace-safe) the user id read on device * num_layers + layer_idx.
    cb_reserve_back(cb_row, 1);
    const uint32_t row_l1 = align64(get_write_ptr(cb_row));
    uint32_t slot = host_slot;
    bool bad_user = false;
    if constexpr (has_batch) {
        const auto batch = TensorAccessor(batch_args, batch_addr);
        noc_async_read(batch.get_noc_addr(0), row_l1, sizeof(uint32_t));
        noc_async_read_barrier();
        const uint32_t user = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row_l1);
        bad_user = user >= num_slots / num_layers;
        slot = bad_user ? 0 : user * num_layers + layer_idx;
    }
    const uint32_t slot_base = slot * pages_per_slot;

    const uint32_t t_p0 = stats_clock();
    // Planner 0 copies this chip's own slice into the inbox, waits until the N - 1 other chips' slices are in (their
    // last packets increment its inbox word), takes the word back to zero (persistent), and releases every planner.
    if (is_sync) {
        cb_reserve_back(cb_inbox, 1);
        const uint32_t buf = align64(get_write_ptr(cb_inbox));
        const auto table = TensorAccessor(table_args, table_addr, inbox_slot_bytes);
        noc_async_read(table.get_noc_addr(0), buf, 64);  // slot 0's header holds this call's extent
        noc_async_read_barrier();
        const uint32_t active =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf)[::sparse_push::plan_table::kWordActiveBlocks];
        noc_async_read(table.get_noc_addr(me), buf, inbox_slot_bytes);
        noc_async_read_barrier();
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf)[::sparse_push::plan_table::kWordActiveBlocks] = active;
        noc_async_write(buf, inbox.get_noc_addr(me), inbox_slot_bytes);
        noc_async_write_barrier();
        auto* inbox_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(inbox_sem_addr);
        noc_semaphore_wait_min(inbox_sem, N - 1);
        noc_semaphore_inc(get_noc_addr(inbox_sem_addr), 0u - (N - 1));
        for (uint32_t p = 0; p < planners; ++p) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(planner_xy_arg + 2 * p),
                    get_arg_val<uint32_t>(planner_xy_arg + 2 * p + 1),
                    go_sem_addr),
                1);
        }
        noc_async_atomic_barrier();
    }
    {
        KVD_ZONE("KVD-P-gowait");
        noc_semaphore_wait_min(go, 1);
    }
    noc_semaphore_set(go, 0);
    const uint32_t t_go = stats_clock();
    KVD_ZONE("KVD-P-plan");

    // Every requester's plan covers its own active blocks (the plan op derives them from that chip's highest requested
    // row), so they differ: a requester's words past its own count are stale and read as zero. The planners split this
    // chip's blocks up to the largest count.
    cb_reserve_back(cb_inbox, 1);
    const uint32_t hdr_l1 = align64(get_write_ptr(cb_inbox));
    for (uint32_t i = 0; i < N; ++i) {
        noc_async_read(inbox.get_noc_addr(i), hdr_l1 + i * 64, 64);
    }
    noc_async_read_barrier();
    uint32_t owned_of[32];
    uint32_t owned = 0;
    for (uint32_t i = 0; i < N; ++i) {
        const uint32_t act = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            hdr_l1 + i * 64)[::sparse_push::plan_table::kWordActiveBlocks];
        owned_of[i] = act > me ? (act - me + N - 1) / N : 0;
        owned_of[i] =
            owned_of[i] < owner_blocks_max ? owned_of[i] : owner_blocks_max;  // a header never overruns a slot
        if (bad_user) {
            owned_of[i] = 0;  // an out-of-range user id: plan nothing rather than read another user's cache
        }
        owned = owned_of[i] > owned ? owned_of[i] : owned;
    }
    // Blocks are dealt to the planners in turn (lb = planner, planner + P, ...): the rows the current chunk's queries
    // pick cluster in a few blocks, so contiguous ranges would leave one planner with most of the work.
    // Slice i's words of the j-th block land at inbox_l1 + (j * N + i) * 128 (one 128 B window covers a block's 5
    // words from its 64 B aligned start).
    const uint32_t inbox_l1 = hdr_l1 + N * 64;
    // Words (32 rows each) are dealt to the planners in turn (u = planner, planner + P, ...): the current chunk's block
    // of this owner is needed by nearly every chip, so even whole blocks leave one planner with most of the work.
    // Slice i's 64 B line holding the j-th word lands at inbox_l1 + (j * N + i) * 64.
    const uint32_t owned_words = owned * words_per_block;
    uint32_t nwords = 0;
    for (uint32_t u = planner; u < owned_words; u += planners, ++nwords) {
        const uint32_t addr = (bitmap_off + u) * 4;
        for (uint32_t i = 0; i < N; ++i) {
            if (owned_of[i] * words_per_block > u) {
                noc_async_read(inbox.get_noc_addr(i) + (addr & ~63u), inbox_l1 + (nwords * N + i) * 64, 64);
            }
        }
    }
    noc_async_read_barrier();

    const uint32_t t_read = stats_clock();
    // Own lists: one staging page per port, flushed to scratch page own_base(port) + page.
    cb_reserve_back(cb_stage, 1);
    const uint32_t stage_l1 = align64(get_write_ptr(cb_stage));
    uint32_t count[kMaxPorts] = {};
    auto stage_page = [&](uint32_t p) { return stage_l1 + p * kScratchPageBytes; };
    auto own_page = [&](uint32_t p, uint32_t k) { return (p * planners + planner) * pages_own + k; };
    auto flush = [&](uint32_t p, bool partial) {
        const uint32_t in_page = count[p] % kEntriesPerPage;
        if (!partial && in_page != 0) {
            return;
        }
        const uint32_t page = (count[p] - (partial ? in_page : kEntriesPerPage)) / kEntriesPerPage;
        const uint32_t bytes = partial ? ((in_page * kEntryBytes + 63) & ~63u) : kScratchPageBytes;
        if (bytes) {
            noc_async_write(stage_page(p), scratch.get_noc_addr(own_page(p, page)), bytes);
            noc_async_write_barrier();
        }
    };

    // The rows this chip itself needs: local copies into its owner-major pages, kCopyBatch at a time.
    constexpr uint32_t kCopyBatch = 8;
    uint32_t copy_rows[kCopyBatch];
    uint32_t copies = 0;
    auto flush_copies = [&]() {
#ifdef KVD_ABL_NO_COPIES
        copies = 0;
        return;
#endif
        for (uint32_t k = 0; k < copies; ++k) {
            noc_async_read(input.get_noc_addr(slot_base + copy_rows[k]), row_l1 + k * row_bytes, row_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t k = 0; k < copies; ++k) {
            noc_async_write(
                row_l1 + k * row_bytes, output.get_noc_addr(me * owner_slot_rows + copy_rows[k]), row_bytes);
        }
        noc_async_writes_flushed();  // the buffers are reused next
        copies = 0;
    };

    const Routing<R, C, WX, WY> rt{kXRing, kYRing};
    const uint32_t me_bit = 1u << me;
    uint32_t j = 0;
#ifdef KVD_ABL_EMPTY_OWN
    for (uint32_t u = owned_words; u < owned_words; u += planners, ++j) {  // no own rows: the bare protocol
#else
    for (uint32_t u = planner; u < owned_words; u += planners, ++j) {
#endif
        const uint32_t lb = u / words_per_block, w = u % words_per_block;
        const uint32_t word_off = ((bitmap_off + u) * 4 % 64) / 4;
        {
            uint32_t words[32];
            uint32_t any = 0;
            for (uint32_t i = 0; i < N; ++i) {
                words[i] = owned_of[i] * words_per_block > u
                               ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(inbox_l1 + (j * N + i) * 64)[word_off]
                               : 0;
                any |= words[i];
            }
            while (any) {
                const uint32_t k = __builtin_ctz(any);
                any &= any - 1;
                uint32_t mask = 0;
                for (uint32_t i = 0; i < N; ++i) {
                    mask |= ((words[i] >> k) & 1) << i;
                }
                const uint32_t in_block = w * 32 + k;
                const uint32_t row = (me + lb * N) * stripe + in_block;
                if (mask & me_bit) {
                    copy_rows[copies++] = lb * stripe + in_block;
                    if (copies == kCopyBatch) {
                        flush_copies();
                    }
                }
                const uint32_t m = mask & ~me_bit;
                if (m == 0) {
                    continue;
                }
                const RowTree<R, C, WX, WY> tree(m);
                const uint32_t order = tree.order(rt, my_r, my_c, row);
                const uint32_t lane = lane_of(row, lanes);
                const uint32_t tie = hbit(row, kTieBit);
                for (uint32_t d = 0; d < kDirections; ++d) {
                    const uint32_t p = port_index(lane, d);
                    if (!(ports_present >> p & 1)) {
                        continue;
                    }
                    const uint32_t hops = tree.own_hops(rt, my_r, my_c, order, d, tie);
                    if (hops == 0) {
                        continue;
                    }
                    auto* e = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        stage_page(p) + (count[p] % kEntriesPerPage) * kEntryBytes);
                    e[0] = entry_word0(row, hops, 0, order);
                    e[1] = m;
                    ++count[p];
                    flush(p, false);
                }
            }
        }
    }
    flush_copies();
    noc_async_write_barrier();
    const uint32_t t_compute = stats_clock();
    // Publish: the partial pages, then each port's list length (64 B per planner in the port's count page), then tell
    // every port core.
    for (uint32_t p = 0; p < kMaxPorts; ++p) {
        if (ports_present >> p & 1) {
            flush(p, true);
        }
    }
    auto* cw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row_l1);
    for (uint32_t p = 0; p < kMaxPorts; ++p) {
        if (!(ports_present >> p & 1)) {
            continue;
        }
        cw[0] = count[p];
        noc_async_write(row_l1, scratch.get_noc_addr(count_base + p) + planner * 64, 64);
        noc_async_write_barrier();
    }
    for (uint32_t p = 0; p < kMaxPorts; ++p) {
        if (ports_present >> p & 1) {
            noc_semaphore_inc(get_noc_addr(port_x[p], port_y[p], own_ready_sem), 1);
        }
    }
    noc_async_atomic_barrier();
#ifdef KVD_STATS
    DPRINT(
        "KVDP planner={} go={} read={} compute={} publish={}\n",
        planner,
        t_go - t_p0,
        t_read - t_go,
        t_compute - t_read,
        stats_clock() - t_compute);
#endif
}
