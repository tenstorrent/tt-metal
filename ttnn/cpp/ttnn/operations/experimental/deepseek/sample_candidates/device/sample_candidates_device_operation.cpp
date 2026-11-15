// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sample_candidates_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::sample_candidates {

namespace {

using namespace tt::tt_metal;

bool is_dram_interleaved(const Tensor& t) {
    return t.memory_config().buffer_type() == BufferType::DRAM &&
           t.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED;
}

bool is_one_element(const Tensor& t) { return t.logical_shape().volume() == 1; }

}  // namespace

void SampleCandidatesDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto& logits = args.logits;
    TT_FATAL(
        logits.storage_type() == StorageType::DEVICE && logits.buffer() != nullptr,
        "decode_sample_candidates: logits must be on device");
    TT_FATAL(
        logits.dtype() == DataType::BFLOAT16 && logits.layout() == Layout::TILE && is_dram_interleaved(logits),
        "decode_sample_candidates: logits must be BFLOAT16 TILE DRAM interleaved");
    const auto& s = logits.logical_shape();
    TT_FATAL(
        s.rank() == 4 && s[0] == 1 && s[1] == 1 && s[2] >= 1 && s[2] <= 8,
        "decode_sample_candidates: logits must be [1, 1, R, V] with 1 <= R <= 8, got {}",
        s);
    TT_FATAL(
        attrs.k >= 1 && attrs.k <= tt::constants::TILE_WIDTH && (2 * attrs.k) % tt::constants::TILE_WIDTH == 0,
        "decode_sample_candidates: k must be 16 or 32, got {}",
        attrs.k);
    const uint32_t tiles = tt::div_up(static_cast<uint32_t>(s[3]), tt::constants::TILE_WIDTH);
    TT_FATAL(
        logits.tensor_spec().tile().get_height() == tt::constants::TILE_HEIGHT &&
            logits.tensor_spec().tile().get_width() == tt::constants::TILE_WIDTH,
        "decode_sample_candidates: logits must be in 32 x 32 tiles");
    TT_FATAL(attrs.num_workers >= 1, "decode_sample_candidates: needs a worker");
    TT_FATAL(
        static_cast<uint32_t>(s[3]) >= attrs.k && tiles >= 1,
        "decode_sample_candidates: V ({}) must be at least k ({})",
        s[3],
        attrs.k);
    TT_FATAL(
        args.inv_temp.dtype() == DataType::BFLOAT16 && args.inv_temp.layout() == Layout::TILE &&
            is_one_element(args.inv_temp) && !args.inv_temp.is_sharded(),
        "decode_sample_candidates: inv_temp must be an interleaved [1, 1, 1, 1] BFLOAT16 TILE");
    TT_FATAL(
        args.index_offset.dtype() == DataType::INT32 && args.index_offset.layout() == Layout::ROW_MAJOR &&
            is_one_element(args.index_offset) && !args.index_offset.is_sharded(),
        "decode_sample_candidates: index_offset must be an interleaved [1, 1, 1, 1] INT32 ROW_MAJOR");
}

void SampleCandidatesDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*args*/) {}

SampleCandidatesDeviceOperation::spec_return_value_t SampleCandidatesDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const ttnn::Shape shape({1, 1, args.logits.logical_shape()[2], 2 * attrs.k});
    return {
        TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), attrs.output_mem_config)),
        TensorSpec(shape, TensorLayout(DataType::INT32, PageConfig(Layout::ROW_MAJOR), attrs.output_mem_config))};
}

SampleCandidatesDeviceOperation::tensor_return_value_t SampleCandidatesDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    tensor_return_value_t outs;
    for (const auto& spec : compute_output_specs(attrs, args)) {
        outs.push_back(create_device_tensor(spec, args.logits.device()));
    }
    return outs;
}

}  // namespace ttnn::operations::experimental::deepseek::sample_candidates

namespace ttnn::prim {

std::vector<Tensor> decode_sample_candidates(
    const Tensor& logits,
    const Tensor& inv_temp,
    const Tensor& index_offset,
    uint32_t k,
    float fill_value,
    uint32_t num_workers) {
    using OperationType = ttnn::operations::experimental::deepseek::sample_candidates::SampleCandidatesDeviceOperation;
    auto attrs = OperationType::operation_attributes_t{
        .k = k,
        .fill_value = fill_value,
        .num_workers = num_workers,
        .output_mem_config = ttnn::DRAM_MEMORY_CONFIG,
    };
    auto tensor_args =
        OperationType::tensor_args_t{.logits = logits, .inv_temp = inv_temp, .index_offset = index_offset};
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
