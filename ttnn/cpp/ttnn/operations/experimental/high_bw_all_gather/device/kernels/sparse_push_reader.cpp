// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PROTOTYPE sparse push reader. Each worker owns a list of streams, one per (this owner -> dest) pair:
//   [dest, dest_base_page, count, local_row_0 .. local_row_{count-1}]
// Rows go into the requester's compact buffer in stream order, packed so that every packet is one contiguous
// bank-lane write: packet j of a stream starts at dest_base_page + (j % banks) + (j / banks) * P * banks and its
// P rows are banks pages apart (interleaved page p lives in bank p % banks). Packets are full whatever the
// locality of the rows, so the transfer cost depends only on the row count.
//
// Stream page layout (one page per link): word 0 = number of streams; stream words start at byte 64. The link's
// four senders (E, W, N, S) share the page; each takes only streams whose first fabric hop leaves in its direction
// (rows for this chip go to the E sender), and skips the rest without reading their rows.

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

#include <cstdint>

namespace {
constexpr uint32_t kData = 0, kMeta = 1, kRuns = 2, kScalar = 3;
constexpr uint32_t kHeaderBytes = 64;
constexpr uint32_t kBatch = 4;  // packets per read barrier
}  // namespace

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t pages_per_packet = get_compile_time_arg_val(1);
    constexpr uint32_t banks = get_compile_time_arg_val(2);
    constexpr uint32_t pages_per_slot = get_compile_time_arg_val(4);
    constexpr bool batch_from_meta = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t block_words = get_compile_time_arg_val(6);
    constexpr uint32_t runs_page_bytes = get_compile_time_arg_val(7);
    constexpr auto input_args = TensorAccessorArgs<8>();
    constexpr auto runs_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto batch_args = TensorAccessorArgs<batch_from_meta ? runs_args.next_compile_time_args_offset() : 0>();

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t runs_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t batch_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t num_layers = get_common_arg_val<uint32_t>(3);
    const uint32_t layer_idx = get_common_arg_val<uint32_t>(4);
    const uint32_t scalar_slot = get_common_arg_val<uint32_t>(5);
    const uint32_t link = get_arg_val<uint32_t>(0);
    const uint32_t dir = get_arg_val<uint32_t>(1);
    const uint32_t my_rank = get_arg_val<uint32_t>(2);
    constexpr uint32_t dest_chip_ids[] = DEST_CHIP_ID;
    constexpr uint32_t dest_mesh_ids[] = DEST_MESH_ID;
    auto mine = [&](uint32_t dest) {
        return dest == my_rank ? dir == 0
                               : static_cast<uint32_t>(tt::tt_fabric::get_next_hop_router_direction(
                                     dest_mesh_ids[dest], dest_chip_ids[dest])) == dir;
    };

    const auto input = TensorAccessor(input_args, input_addr, page_bytes);
    const auto runs = TensorAccessor(runs_args, runs_addr, runs_page_bytes);

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

    // Word stream over this worker's page, refilled one block at a time.
    cb_reserve_back(kRuns, 1);
    const uint32_t block_l1 = get_write_ptr(kRuns);
    const uint64_t page_noc = runs.get_noc_addr(link);
    uint32_t block_start = 0, cursor = 0;  // word offsets into the page
    auto fill = [&](uint32_t word) {
        block_start = word & ~15u;  // 64 B aligned
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
    const uint32_t num_streams = next_word();
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
    for (uint32_t s = 0; s < num_streams; ++s) {
        const uint32_t dest = next_word();
        const uint32_t base = next_word();
        const uint32_t count = next_word();
        if (!mine(dest)) {
            cursor += count;
            continue;
        }
        for (uint32_t j = 0; j * pages_per_packet < count; ++j) {
            const uint32_t n =
                count - j * pages_per_packet < pages_per_packet ? count - j * pages_per_packet : pages_per_packet;
            cb_reserve_back(kData, batched + 1);
            cb_reserve_back(kMeta, batched + 1);
            const uint32_t data_l1 = get_write_ptr(kData) + batched * pages_per_packet * page_bytes;
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t row = next_word();
#ifndef SPARSE_PUSH_SKIP
                noc_async_read(input.get_noc_addr(slot_base + row), data_l1 + i * page_bytes, page_bytes);
#endif
            }
            auto* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kMeta) + batched * 16);
            meta[0] = dest;
            meta[1] = base + (j % banks) + (j / banks) * pages_per_packet * banks;
            meta[2] = n;
            if (++batched == kBatch) {
                flush_batch();
            }
        }
    }
    flush_batch();
    // Terminator.
    cb_reserve_back(kData, 1);
    cb_reserve_back(kMeta, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kMeta))[2] = 0;
    cb_push_back(kData, 1);
    cb_push_back(kMeta, 1);
}
