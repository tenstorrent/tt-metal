// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PROTOTYPE multicast-tree push reader for fixed-size records (SPARSE_PUSH_TREE=2), the format the device tree
// planner writes (reference: _tree_plan_records in tests/nightly/blackhole/sdpa/test_sparse_sdpa_gather_overlap.py).
// Same protocol as tree_push_reader.cpp; one sender page holds words 1..16 = packets this chip receives per neighbour
// sender, then records from word 32.
// kind 1/3: src = input row (to a neighbour / to this chip), 2: src = page of this chip's output, 6: level end, 0:
// end. Records are 128 B: [kind, target, n, 0 x 5, (src, arrival) x P, 0...]. A row with arrival = stream << 24 |
// count is read once that neighbour sender has delivered count packets.

#include "api/dataflow/dataflow_api.h"

#include <cstdint>

namespace {
constexpr uint32_t kData = 0, kMeta = 1, kRuns = 2, kScalar = 3, kCounters = 5;
constexpr uint32_t kHeaderWords = 32;
constexpr uint32_t kStreams = 16;
constexpr uint32_t kBatch = 4;  // packets per read barrier
constexpr uint32_t kSlots = 8;  // kData/kMeta depth; a batch's slots are addressed linearly, so it must not wrap
enum : uint32_t { kEnd = 0, kSendInput = 1, kSendOutput = 2, kLocal = 3, kLevel = 6 };
}  // namespace

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t pages_per_packet = get_compile_time_arg_val(1);
    constexpr uint32_t pages_per_slot = get_compile_time_arg_val(2);
    constexpr bool batch_from_meta = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t block_words = get_compile_time_arg_val(4);
    constexpr uint32_t runs_page_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t go_sem_id = get_compile_time_arg_val(6);
    constexpr auto input_args = TensorAccessorArgs<7>();
    constexpr auto runs_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<runs_args.next_compile_time_args_offset()>();
    constexpr auto batch_args = TensorAccessorArgs<batch_from_meta ? output_args.next_compile_time_args_offset() : 0>();

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t runs_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t batch_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t num_layers = get_common_arg_val<uint32_t>(3);
    const uint32_t layer_idx = get_common_arg_val<uint32_t>(4);
    const uint32_t scalar_slot = get_common_arg_val<uint32_t>(5);
    const uint32_t output_addr = get_common_arg_val<uint32_t>(7);
    const uint32_t worker = get_arg_val<uint32_t>(0);
    const uint32_t sync_x = get_arg_val<uint32_t>(1);
    const uint32_t sync_y = get_arg_val<uint32_t>(2);

    const auto input = TensorAccessor(input_args, input_addr, page_bytes);
    const auto runs = TensorAccessor(runs_args, runs_addr, runs_page_bytes);
    const auto output = TensorAccessor(output_args, output_addr, page_bytes);

    cb_reserve_back(kScalar, 1);
    const uint32_t scalar_l1 = get_write_ptr(kScalar);
    uint32_t slot = scalar_slot;
    if constexpr (batch_from_meta) {
        const auto batch = TensorAccessor(batch_args, batch_addr);
        noc_async_read(batch.get_noc_addr(0), scalar_l1, sizeof(uint32_t));
        noc_async_read_barrier();
        invalidate_l1_cache();
        slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scalar_l1)[0] * num_layers + layer_idx;
    }
    const uint32_t slot_base = slot * pages_per_slot;

    cb_reserve_back(kRuns, 1);
    const uint32_t block_l1 = get_write_ptr(kRuns);
    const uint64_t page_noc = runs.get_noc_addr(worker);
    uint32_t block_start = 0, cursor = 0;
    auto fill = [&](uint32_t word) {
        block_start = word & ~15u;
        constexpr uint32_t page_words = runs_page_bytes / 4;
        const uint32_t words = page_words - block_start < block_words ? page_words - block_start : block_words;
        noc_async_read(page_noc + block_start * 4, block_l1, words * 4);
        noc_async_read_barrier();
        invalidate_l1_cache();
    };
    auto next_word = [&]() {
        if (cursor < block_start || cursor >= block_start + block_words) {
            fill(cursor);
        }
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(block_l1)[cursor++ - block_start];
    };

    fill(0);
    uint32_t expected[kStreams];
    cursor = 1;
    for (uint32_t s = 0; s < kStreams; ++s) {
        expected[s] = next_word();
    }
    cursor = kHeaderWords;

    const uint32_t counters_l1 = get_write_ptr(kCounters);
    const uint64_t remote_counters = get_noc_addr(sync_x, sync_y, counters_l1);
    // All counters at once: an L1-to-L1 read needs matching 16 B offsets at both ends.
    auto* poll = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scalar_l1);
    uint32_t seen[kStreams] = {};
    auto* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(go_sem_id));
    noc_semaphore_wait_min(go, 1);
    noc_semaphore_inc(get_noc_addr(get_semaphore(go_sem_id)), uint32_t{0} - 1u);
    noc_async_atomic_barrier();

    uint32_t batched = 0, wpos = 0;  // wpos: ring slot of the batch's first packet
    auto flush_batch = [&]() {
        if (batched) {
            noc_async_read_barrier();
            cb_push_back(kData, batched);
            cb_push_back(kMeta, batched);
            wpos = (wpos + batched) % kSlots;
            batched = 0;
        }
    };
    auto push_marker = [&](uint32_t kind) {
        flush_batch();
        cb_reserve_back(kData, 1);
        cb_reserve_back(kMeta, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kMeta))[0] = kind;
        cb_push_back(kData, 1);
        cb_push_back(kMeta, 1);
        wpos = (wpos + 1) % kSlots;
    };

    constexpr uint32_t record_words = 32;
    for (uint32_t rec = kHeaderWords;; rec += record_words) {
        cursor = rec;
        const uint32_t type = next_word();
        if (type == kEnd) {
            push_marker(kEnd);
            break;
        }
        if (type == kLevel) {
            push_marker(kLevel);
            continue;
        }
        const uint32_t target = next_word();
        const uint32_t n = next_word();
        cursor = rec + 8;
        cb_reserve_back(kData, batched + 1);
        cb_reserve_back(kMeta, batched + 1);
        const uint32_t data_l1 = get_write_ptr(kData) + batched * pages_per_packet * page_bytes;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t src = next_word();
            const uint32_t arrival = next_word();
            if (type == kSendOutput && arrival != 0) {
                const uint32_t stream = arrival >> 24, count = arrival & 0xFFFFFF;
                if (seen[stream] < count) {
                    flush_batch();  // hand finished packets to the writer before blocking
                    while (seen[stream] < count) {
                        noc_async_read(remote_counters, reinterpret_cast<uint32_t>(poll), kStreams * 4);
                        noc_async_read_barrier();
                        invalidate_l1_cache();
                        for (uint32_t s = 0; s < kStreams; ++s) {
                            seen[s] = poll[s];
                        }
                    }
                }
            }
            const uint64_t addr = type == kSendOutput ? output.get_noc_addr(src) : input.get_noc_addr(slot_base + src);
            noc_async_read(addr, data_l1 + i * page_bytes, page_bytes);
        }
        auto* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kMeta) + batched * 16);
        meta[0] = type;
        meta[1] = target;
        meta[2] = n;
        if (++batched == kBatch || wpos + batched == kSlots) {
            flush_batch();
        }
    }

    if (worker == 0) {
        // Every packet bound for this chip has landed once each neighbour sender's count is reached.
        auto* counters = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counters_l1);
        for (uint32_t s = 0; s < kStreams; ++s) {
            while (counters[s] < expected[s]) {
                invalidate_l1_cache();
            }
        }
    }
}
