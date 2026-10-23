// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::high_bw_all_gather {

// On-device plan for the sparse KV push. From one requester chip's top-k indices (natural positions in the
// TP-deduped block-cyclic KVPE cache) it builds
//   - the requester's plan table: which rows it needs (bitmap) plus the counts every owner and the requester itself
//     derive the compact-buffer layout from; all-gathered, it tells each owner what to push and where; and
//   - the indices rewritten to slots in the requester's compact buffer.
// Natural row n lives in block n / stripe_chunk, owned by chip block % num_devices. A requester's rows from owner o
// occupy a run starting at base(o) = sum over o' < o of round_up(count(o'), P * banks); row k of the run sits in the
// bank lane (k / P) % banks of packet group (k / P) / banks, at offset k % P (see
// kernels/sparse_push_table_reader.cpp).
//
// Table: uint32 [1, 1, pages, 1024], one 4 KiB row-major page per row. Word offsets:
//   [0] magic, [1] num_devices, [2] stripe_chunk, [3] active blocks (later blocks are stale), [4] split block
//   [32, 32 + num_devices)   rows this requester needs from each owner
//   [64, 64 + num_devices)   of those, rows in blocks before the split block (the chunk start): the prefix part
//   [128, 128 + W)           bitmap of needed rows, W = max_rows / 32
// It holds only what the owners' push reads, since every chip all-gathers every table. The plan op's own image of it
// carries the block counts after the bitmap (counts_off), which stay on chip.
struct SparsePushPlanLayout {
    static constexpr uint32_t kMagic = 0x5350504Cu;
    static constexpr uint32_t kOwnerOff = 32;
    static constexpr uint32_t kOwnerPrefixOff = 64;
    static constexpr uint32_t kBitmapOff = 128;
    static constexpr uint32_t kPageWords = 1024;

    uint32_t bitmap_words = 0;
    uint32_t num_blocks = 0;
    uint32_t counts_off = 0;   // words, in the plan op's image only
    uint32_t image_words = 0;  // the plan op's image: table words, then the block counts
    uint32_t pages = 0;

    static SparsePushPlanLayout make(uint32_t max_rows, uint32_t stripe_chunk) {
        auto round_up = [](uint32_t x, uint32_t m) { return (x + m - 1) / m * m; };
        SparsePushPlanLayout l;
        l.bitmap_words = max_rows / 32;
        l.num_blocks = max_rows / stripe_chunk;
        l.pages = round_up(kBitmapOff + l.bitmap_words, kPageWords) / kPageWords;
        l.counts_off = round_up(kBitmapOff + l.bitmap_words, 16);
        l.image_words = std::max(l.pages * kPageWords, l.counts_off + l.num_blocks);
        return l;
    }
};

struct SparsePushPlanParams {
    uint32_t num_devices = 0;
    uint32_t stripe_chunk = 0;
    uint32_t max_rows = 0;
    uint32_t pages_per_packet = 0;
    uint32_t banks = 0;
};

struct SparsePushPlanInputs {
    Tensor indices;  // uint32 row-major [1, 1, rows, topk] per device, 0xFFFFFFFF = masked
    // Optional 1-element uint32 DRAM tensor holding the chunk start row: rows before it are the prefix part.
    std::optional<Tensor> chunk_start;
};

struct SparsePushPlanDeviceOperation {
    using operation_attributes_t = SparsePushPlanParams;
    using tensor_args_t = SparsePushPlanInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;  // {table, compact indices}

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& attrs, const tensor_args_t& args, tensor_return_value_t& outputs);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::high_bw_all_gather

namespace ttnn::prim {

std::vector<Tensor> sparse_push_plan(
    const Tensor& indices,
    uint32_t num_devices,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    uint32_t pages_per_packet,
    uint32_t banks,
    const std::optional<Tensor>& chunk_start = std::nullopt);

}  // namespace ttnn::prim
