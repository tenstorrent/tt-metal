// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PROTOTYPE one-hop relay variant of the sparse push (selected with SPARSE_RELAY=1). Every packet moves one torus
// hop at a time: vertical (N/S) first to the destination's row, then horizontal (E/W). Intermediate chips stage a
// packet in one P-row bank lane of a staging region (the tail of the same output buffer) and forward it unchanged;
// the last hop writes the bank lane in the destination's compact buffer that the routed variant would write, so
// the slot layout (see sparse_push_reader.cpp) is identical.
//
// Each sender (link, direction) has its own host-built stream page:
//   word 0 = number of rounds R; items start at word 16:
//     [0]                              end of round
//     [1, target_page, n, row_0..n-1]  n input rows -> neighbour page target_page (bank lane)
//     [2, target_page, n, src_page]    n staged rows (bank lane at src_page) -> neighbour page target_page
//     [3, target_page, n, row_0..n-1]  n input rows -> this chip's page target_page
// Round h carries hop h of every packet, so a packet staged here in round h-1 is forwarded in round h. Before
// reading round h's staged packets the reader waits until every neighbour sender pointing at this chip has
// finished round h-1 (each one increments done on every worker core of this chip after its round's data).

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

#include <cstdint>

namespace {
constexpr uint32_t kData = 0, kMeta = 1, kRuns = 2, kScalar = 3;
constexpr uint32_t kHeaderBytes = 64;
constexpr uint32_t kBatch = 4;  // packets per read barrier
constexpr uint32_t kRoundEnd = 0, kTerminate = 4;
}  // namespace

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t pages_per_packet = get_compile_time_arg_val(1);
    constexpr uint32_t pages_per_slot = get_compile_time_arg_val(2);
    constexpr bool batch_from_meta = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t block_words = get_compile_time_arg_val(4);
    constexpr uint32_t runs_page_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t per_round = get_compile_time_arg_val(6);  // done increments per round on each core
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
    const uint32_t output_addr = get_common_arg_val<uint32_t>(6);
    const uint32_t done_sem = get_common_arg_val<uint32_t>(7);
    const uint32_t page = get_arg_val<uint32_t>(0);  // this sender's stream page

    const auto input = TensorAccessor(input_args, input_addr, page_bytes);
    const auto runs = TensorAccessor(runs_args, runs_addr, runs_page_bytes);
    const auto output = TensorAccessor(output_args, output_addr, page_bytes);
    auto* done_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_sem);

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
    const uint64_t page_noc = runs.get_noc_addr(page);
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
    const uint32_t rounds = next_word();
    cursor = kHeaderBytes / 4;

    uint32_t batched = 0;
    auto flush_batch = [&]() {
        if (batched) {
            noc_async_read_barrier();
            cb_push_back(kData, batched);
            cb_push_back(kMeta, batched);
            batched = 0;
        }
    };
    auto push_marker = [&](uint32_t kind, uint32_t arg) {
        cb_reserve_back(kData, 1);
        cb_reserve_back(kMeta, 1);
        auto* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kMeta));
        meta[0] = kind;
        meta[2] = arg;
        cb_push_back(kData, 1);
        cb_push_back(kMeta, 1);
    };

    uint32_t round = 0;
    while (round < rounds) {
        const uint32_t type = next_word();
        if (type == kRoundEnd) {
            flush_batch();
            push_marker(kRoundEnd, 0);
            ++round;
            if (round < rounds) {
                // Staged packets for the next round must have landed here.
                noc_semaphore_wait_min(done_ptr, per_round * round);
            }
            continue;
        }
        const uint32_t target = next_word();
        const uint32_t n = next_word();
        cb_reserve_back(kData, batched + 1);
        cb_reserve_back(kMeta, batched + 1);
        const uint32_t data_l1 = get_write_ptr(kData) + batched * pages_per_packet * page_bytes;
        if (type == 2) {
            const uint32_t src = next_word();
            noc_async_read(output.get_noc_addr(src), data_l1, n * page_bytes);
        } else {
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t row = next_word();
                noc_async_read(input.get_noc_addr(slot_base + row), data_l1 + i * page_bytes, page_bytes);
            }
        }
        auto* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kMeta) + batched * 16);
        meta[0] = type;
        meta[1] = target;
        meta[2] = n;
        if (++batched == kBatch) {
            flush_batch();
        }
    }
    flush_batch();
    push_marker(kTerminate, rounds);
}
