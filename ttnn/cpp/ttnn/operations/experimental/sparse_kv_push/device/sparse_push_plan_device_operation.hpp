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
#include "kernels/sparse_push_constants.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

// Compact slots: owner o's rows start at base(o) = sum over o' < o of round_up(count(o'), P * banks). Row k sits in
// bank lane (k / P) % banks at offset k % P, so each P-row packet is one contiguous write into a single DRAM bank.
//
// Table words: see ::sparse_push::plan_table in kernels/sparse_push_constants.hpp.
struct SparsePushPlanLayout {
    static constexpr uint32_t kBitmapOff = ::sparse_push::plan_table::kBitmapOff;
    static constexpr uint32_t kPageWords = ::sparse_push::plan_table::kPageWords;

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
    uint32_t page_bytes = 0;  // the push's cache row page; sets the packet shape the compact slots follow
};

struct SparsePushPlanInputs {
    Tensor indices;  // uint32 row-major [1, 1, rows, topk] per device, 0xFFFFFFFF = masked
    // 1-element uint32 chunk start row: rows before it are the prefix part.
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

}  // namespace ttnn::operations::experimental::sparse_kv_push

namespace ttnn::prim {

std::vector<Tensor> sparse_push_plan(
    const Tensor& indices,
    uint32_t num_devices,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    uint32_t page_bytes,
    const std::optional<Tensor>& chunk_start = std::nullopt);

}  // namespace ttnn::prim
