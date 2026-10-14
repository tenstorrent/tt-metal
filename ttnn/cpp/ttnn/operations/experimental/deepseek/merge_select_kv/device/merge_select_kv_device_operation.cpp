// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "merge_select_kv_device_operation.hpp"

#include <tt_stl/assert.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::merge_select_kv {

using namespace tt::tt_metal;

MergeSelectKvDeviceOperation::program_factory_t MergeSelectKvDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ProgramFactory{};
}

void MergeSelectKvDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TT_FATAL(args.k > 0, "merge_select_kv: k must be positive");
    const auto& candidates = tensor_args.candidates;
    TT_FATAL(candidates.storage_type() == StorageType::DEVICE, "merge_select_kv: candidates must be on device");
    TT_FATAL(
        candidates.dtype() == DataType::INT32 && candidates.layout() == Layout::ROW_MAJOR &&
            candidates.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "merge_select_kv: candidates must be INT32 row-major interleaved");
    const auto& cand_shape = candidates.logical_shape();
    TT_FATAL(
        cand_shape.rank() == 4 && cand_shape[0] == 1 && cand_shape[1] == 1 && cand_shape[-1] == 2 * args.k + 16,
        "merge_select_kv: candidates must be [1, 1, num_parts, {}] for k = {}, got {}",
        2 * args.k + 16,
        args.k,
        cand_shape);

    const auto& kv_cache = tensor_args.kv_cache;
    TT_FATAL(
        kv_cache.storage_type() == StorageType::DEVICE && kv_cache.logical_shape().rank() == 4,
        "merge_select_kv: kv_cache must be a rank-4 device tensor");
    TT_FATAL(
        kv_cache.layout() == Layout::ROW_MAJOR &&
            kv_cache.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "merge_select_kv: kv_cache must be row-major interleaved (one row per page)");
    const auto& kv_shape = kv_cache.logical_shape();
    TT_FATAL(kv_shape[1] == 1, "merge_select_kv: kv_cache heads must be 1, got {}", kv_shape[1]);
    TT_FATAL(
        args.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "merge_select_kv: the output must be interleaved, got {}",
        args.output_mem_config);

    if (tensor_args.output_tensor.has_value()) {
        const auto& out = tensor_args.output_tensor.value();
        TT_FATAL(out.storage_type() == StorageType::DEVICE, "merge_select_kv: output_tensor must be on device");
        const auto& out_shape = out.logical_shape();
        TT_FATAL(
            out.dtype() == kv_cache.dtype() && out.layout() == Layout::ROW_MAJOR &&
                out.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "merge_select_kv: output_tensor must be row-major interleaved with kv_cache's dtype {}",
            kv_cache.dtype());
        TT_FATAL(
            out_shape.rank() == 4 && out_shape[0] == 1 && out_shape[1] == 1 && out_shape[-1] == kv_shape[-1],
            "merge_select_kv: output_tensor must be [1, 1, rows, {}], got {}",
            kv_shape[-1],
            out_shape);
        TT_FATAL(
            args.output_row_offset + args.k <= out_shape[-2],
            "merge_select_kv: output_row_offset {} + k {} exceeds output_tensor rows {}",
            args.output_row_offset,
            args.k,
            out_shape[-2]);
    } else {
        TT_FATAL(
            args.output_row_offset == 0,
            "merge_select_kv: output_row_offset {} needs an output_tensor",
            args.output_row_offset);
    }
}

MergeSelectKvDeviceOperation::spec_return_value_t MergeSelectKvDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor->tensor_spec();
    }
    const auto& kv = tensor_args.kv_cache;
    return TensorSpec(
        ttnn::Shape({1, 1, args.k, kv.logical_shape()[-1]}),
        TensorLayout(kv.dtype(), PageConfig(Layout::ROW_MAJOR), args.output_mem_config));
}

MergeSelectKvDeviceOperation::tensor_return_value_t MergeSelectKvDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value();
    }
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.kv_cache.device());
}

}  // namespace ttnn::operations::experimental::deepseek::merge_select_kv

namespace ttnn::prim {

ttnn::Tensor merge_select_kv(
    const ttnn::Tensor& candidates,
    const ttnn::Tensor& kv_cache,
    uint32_t k,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& output_tensor,
    uint32_t output_row_offset) {
    using OperationType = ttnn::operations::experimental::deepseek::merge_select_kv::MergeSelectKvDeviceOperation;
    auto attrs = OperationType::operation_attributes_t{
        .k = k,
        .output_row_offset = output_row_offset,
        // SDPA reads the gathered rows next, so they stay in L1 by default.
        .output_mem_config = memory_config.value_or(
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1}),
    };
    auto tensor_args = OperationType::tensor_args_t{
        .candidates = candidates,
        .kv_cache = kv_cache,
        .output_tensor = output_tensor,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
