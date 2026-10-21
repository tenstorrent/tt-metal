// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/deepseek_v4_prefill_indexer_select_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/deepseek_v4_prefill_indexer_select_common.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <algorithm>

namespace ttnn::prim {

namespace {
namespace dis = ::deepseek_v4_prefill_indexer_select;

void check_dram(const Tensor& t, Layout layout, const char* name) {
    TT_FATAL(
        t.storage_type() == StorageType::DEVICE && t.buffer() != nullptr,
        "deepseek_v4_prefill_indexer_select: {} must be on device",
        name);
    TT_FATAL(t.layout() == layout, "deepseek_v4_prefill_indexer_select: {} must be {}", name, layout);
    TT_FATAL(
        t.memory_config().buffer_type() == BufferType::DRAM,
        "deepseek_v4_prefill_indexer_select: {} must be in DRAM",
        name);
    TT_FATAL(!t.memory_config().is_sharded(), "deepseek_v4_prefill_indexer_select: {} must be interleaved", name);
}

// What the program hash does not key on, checked on every dispatch: placement, layout and the entry count E.
void validate_non_hashed(const DeepseekV4PrefillIndexerSelectInputs& t) {
    const auto* device = t.q.device();
    check_dram(t.q, Layout::TILE, "q");
    check_dram(t.weights, Layout::TILE, "weights");
    check_dram(t.idx_keys, Layout::ROW_MAJOR, "idx_keys");
    check_dram(t.start, Layout::ROW_MAJOR, "start");
    TT_FATAL(t.weights.device() == device, "deepseek_v4_prefill_indexer_select: weights must be on q's device");
    TT_FATAL(t.idx_keys.device() == device, "deepseek_v4_prefill_indexer_select: idx_keys must be on q's device");
    TT_FATAL(t.start.device() == device, "deepseek_v4_prefill_indexer_select: start must be on q's device");
    TT_FATAL(t.start.logical_volume() == 1, "deepseek_v4_prefill_indexer_select: start must hold exactly one element");
    const auto ks = t.idx_keys.logical_shape();
    TT_FATAL(
        ks.rank() == 4 && ks[0] == 1 && ks[1] == 1 && ks[2] > 0 && ks[3] == t.q.logical_shape()[3],
        "deepseek_v4_prefill_indexer_select: idx_keys must be [1, 1, E, {}] with E > 0 (got {})",
        t.q.logical_shape()[3],
        ks);
    TT_FATAL(t.idx_keys.padded_shape() == ks, "deepseek_v4_prefill_indexer_select: idx_keys must not be padded");
}
}  // namespace

uint32_t deepseek_v4_prefill_indexer_select_num_splits(uint32_t num_q_tiles, uint32_t num_cores) {
    return std::clamp<uint32_t>(num_cores / num_q_tiles, 1, dis::MAX_SPLITS);
}

void DeepseekV4PrefillIndexerSelectOperation::validate_on_program_cache_hit(
    const DeepseekV4PrefillIndexerSelectParams& /*attrs*/, const DeepseekV4PrefillIndexerSelectInputs& t) {
    validate_non_hashed(t);
}

void DeepseekV4PrefillIndexerSelectOperation::validate_on_program_cache_miss(
    const DeepseekV4PrefillIndexerSelectParams& attrs, const DeepseekV4PrefillIndexerSelectInputs& t) {
    TT_FATAL(
        tt::tt_metal::hal::get_arch() == tt::ARCH::BLACKHOLE,
        "deepseek_v4_prefill_indexer_select is Blackhole-only (TopK XL)");
    TT_FATAL(t.q.dtype() == DataType::BFLOAT16, "deepseek_v4_prefill_indexer_select: q must be bf16");
    TT_FATAL(t.weights.dtype() == DataType::BFLOAT16, "deepseek_v4_prefill_indexer_select: weights must be bf16");
    TT_FATAL(t.idx_keys.dtype() == DataType::BFLOAT16, "deepseek_v4_prefill_indexer_select: idx_keys must be bf16");
    TT_FATAL(
        t.start.dtype() == DataType::UINT32 || t.start.dtype() == DataType::INT32,
        "deepseek_v4_prefill_indexer_select: start must be uint32 or int32");
    validate_non_hashed(t);

    constexpr uint32_t tile = tt::constants::TILE_WIDTH;
    const auto qs = t.q.logical_shape();
    TT_FATAL(qs.rank() == 4 && qs[0] == 1, "deepseek_v4_prefill_indexer_select: q must be [1, H, T, D] (got {})", qs);
    const uint32_t H = qs[1];
    const uint32_t T = qs[2];
    const uint32_t D = qs[3];
    TT_FATAL(
        H > 0 && H % tile == 0,
        "deepseek_v4_prefill_indexer_select: H must be a positive multiple of {} (got {})",
        tile,
        H);
    TT_FATAL(
        T > 0 && T % tile == 0,
        "deepseek_v4_prefill_indexer_select: T must be a positive multiple of {} (got {})",
        tile,
        T);
    TT_FATAL(
        D > 0 && D % tile == 0,
        "deepseek_v4_prefill_indexer_select: D must be a positive multiple of {} (got {})",
        tile,
        D);
    TT_FATAL(
        t.weights.logical_shape() == ttnn::Shape({1, 1, T, H}),
        "deepseek_v4_prefill_indexer_select: weights must be [1, 1, {}, {}] (got {})",
        T,
        H,
        t.weights.logical_shape());
    TT_FATAL(attrs.compress_ratio > 0, "deepseek_v4_prefill_indexer_select: compress_ratio must be > 0");
    TT_FATAL(
        (D * sizeof(uint16_t)) % tt::tt_metal::hal::get_dram_alignment() == 0,
        "deepseek_v4_prefill_indexer_select: idx_keys row bytes must be DRAM aligned");
}

DeepseekV4PrefillIndexerSelectOperation::spec_return_value_t
DeepseekV4PrefillIndexerSelectOperation::compute_output_specs(
    const DeepseekV4PrefillIndexerSelectParams& attrs, const DeepseekV4PrefillIndexerSelectInputs& t) {
    const uint32_t T = t.q.logical_shape()[2];
    const auto grid = t.q.device()->compute_with_storage_grid_size();
    const uint32_t splits =
        deepseek_v4_prefill_indexer_select_num_splits(T / tt::constants::TILE_HEIGHT, grid.x * grid.y);
    const tt::tt_metal::MemoryConfig dram{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    const auto rm = [&](const ttnn::Shape& shape) {
        return tt::tt_metal::TensorSpec(
            shape, tt::tt_metal::TensorLayout(DataType::UINT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), dram));
    };
    // Scratch page (peer, query, value / index) holds one fp32 tile.
    const uint32_t scratch_pages = attrs.candidates ? 1 : std::max<uint32_t>(1, (splits - 1) * T * 2);
    // Candidate scratch page (split, query, survivor tile) holds one fp32 tile.
    const uint32_t rows = attrs.candidates ? 1 : T;
    const uint32_t cand_rows = attrs.candidates ? T : 1;
    const uint32_t cand_scratch_pages = attrs.candidates ? splits * T * dis::CAND_SURV_TILES : 1;
    return {
        rm(ttnn::Shape({1, 1, rows, dis::TOPK})),
        rm(ttnn::Shape({1, 1, scratch_pages, tt::constants::TILE_HW})),
        rm(ttnn::Shape({1, 1, cand_rows, dis::CAND_TOPK})),
        rm(ttnn::Shape({1, 1, cand_scratch_pages, tt::constants::TILE_HW}))};
}

DeepseekV4PrefillIndexerSelectOperation::tensor_return_value_t
DeepseekV4PrefillIndexerSelectOperation::create_output_tensors(
    const DeepseekV4PrefillIndexerSelectParams& attrs, const DeepseekV4PrefillIndexerSelectInputs& t) {
    const auto specs = compute_output_specs(attrs, t);
    auto* device = t.q.device();
    return {
        create_device_tensor(specs[0], device),
        create_device_tensor(specs[1], device),
        create_device_tensor(specs[2], device),
        create_device_tensor(specs[3], device)};
}

ttsl::hash::hash_t DeepseekV4PrefillIndexerSelectOperation::compute_program_hash(
    const DeepseekV4PrefillIndexerSelectParams& attrs, const DeepseekV4PrefillIndexerSelectInputs& t) {
    // idx_keys' entry count E rides on a runtime arg, so a longer buffer reuses the program.
    return tt::tt_metal::operation::hash_operation<DeepseekV4PrefillIndexerSelectOperation>(
        attrs.compress_ratio, attrs.candidates, attrs.compute_kernel_config, t.q.logical_shape(), t.start.dtype());
}

Tensor deepseek_v4_prefill_indexer_select(
    const Tensor& q,
    const Tensor& weights,
    const Tensor& idx_keys,
    const Tensor& start,
    uint32_t compress_ratio,
    bool candidates,
    ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    using OperationType = ttnn::prim::DeepseekV4PrefillIndexerSelectOperation;
    auto outputs = ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .compress_ratio = compress_ratio,
            .candidates = candidates,
            .compute_kernel_config = compute_kernel_config,
        },
        OperationType::tensor_args_t{
            .q = q,
            .weights = weights,
            .idx_keys = idx_keys,
            .start = start,
        });
    return candidates ? outputs[2] : outputs[0];
}

}  // namespace ttnn::prim
