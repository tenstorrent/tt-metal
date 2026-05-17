// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_unary_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::device {

RoutedUnaryDeviceOperation::program_factory_t RoutedUnaryDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return RoutedUnaryProgramFactory{};
}

void RoutedUnaryDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    TT_FATAL(tensor_args.input.storage_type() == StorageType::DEVICE, "routed_unary: input must be on device");
    TT_FATAL(
        tensor_args.global_expert_idx_table.storage_type() == StorageType::DEVICE,
        "routed_unary: global_expert_idx_table must be on device");
    TT_FATAL(
        tensor_args.global_expert_idx_table.buffer() != nullptr &&
            tensor_args.global_expert_idx_table.buffer()->buffer_type() == BufferType::DRAM,
        "routed_unary: global_expert_idx_table must be in DRAM");
    TT_FATAL(
        tensor_args.expert_token_counts.storage_type() == StorageType::DEVICE,
        "routed_unary: expert_token_counts must be on device");
    TT_FATAL(
        tensor_args.expert_token_counts.buffer() != nullptr &&
            tensor_args.expert_token_counts.buffer()->buffer_type() == BufferType::DRAM,
        "routed_unary: expert_token_counts must be in DRAM");
}

void RoutedUnaryDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(operation_attributes, tensor_args);

    TT_FATAL(tensor_args.input.layout() == tt::tt_metal::Layout::TILE, "routed_unary: only TILE layout is supported");
    TT_FATAL(tensor_args.input.is_sharded(), "routed_unary: only sharded input is supported");
    TT_FATAL(operation_attributes.output_memory_config.is_sharded(), "routed_unary: only sharded output is supported");
}

RoutedUnaryDeviceOperation::spec_return_value_t RoutedUnaryDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }
    const auto& input = tensor_args.input;
    return tt::tt_metal::TensorSpec(
        input.logical_shape(),
        tt::tt_metal::TensorLayout(
            operation_attributes.output_dtype,
            tt::tt_metal::PageConfig(input.layout()),
            operation_attributes.output_memory_config));
}

RoutedUnaryDeviceOperation::tensor_return_value_t RoutedUnaryDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input.device());
}

ttsl::hash::hash_t RoutedUnaryDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    // Exclude local_expert_idx / curr_expert_iter / expert_iter_length: these
    // are runtime-arg values only. Include everything that could affect kernel
    // compilation or program layout.
    return ttsl::hash::hash_objects_with_default_seed(
        attrs.op_chain,
        attrs.compute_kernel_config,
        attrs.output_memory_config,
        attrs.output_dtype,
        attrs.fp32_dest_acc_en,
        attrs.preserve_fp32_precision,
        attrs.bfp8_pack_precise,
        args.input.tensor_spec(),
        args.global_expert_idx_table.tensor_spec(),
        args.expert_token_counts.tensor_spec(),
        args.optional_output_tensor.has_value());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::device

namespace ttnn::prim {

ttnn::Tensor routed_unary(
    const ttnn::Tensor& input,
    const ttnn::operations::unary::EltwiseUnaryWithParam& op,
    const ttnn::Tensor& global_expert_idx_table,
    const ttnn::Tensor& expert_token_counts,
    uint32_t local_expert_idx,
    uint32_t curr_expert_iter,
    uint32_t expert_iter_length,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config,
    std::optional<ttnn::Tensor> optional_output_tensor,
    const std::optional<tt::tt_metal::DataType>& output_dtype) {
    namespace ru = ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::device;
    using OperationType = ru::RoutedUnaryDeviceOperation;

    const tt::tt_metal::DataType out_dtype = output_dtype.value_or(input.dtype());

    const tt::tt_metal::MemoryConfig resolved_output_mem_cfg =
        output_memory_config.has_value()
            ? output_memory_config.value()
            : (optional_output_tensor.has_value() ? optional_output_tensor->memory_config() : input.memory_config());

    // Mirror unary_ng's heuristics for fp32 dest accumulation + precision flags.
    const bool preserve_fp32_precision = (input.dtype() == tt::tt_metal::DataType::FLOAT32);
    const bool fp32_dest_acc_en =
        preserve_fp32_precision || out_dtype == tt::tt_metal::DataType::UINT32 ||
        out_dtype == tt::tt_metal::DataType::INT32 || out_dtype == tt::tt_metal::DataType::FLOAT32 ||
        out_dtype == tt::tt_metal::DataType::UINT8 || input.dtype() == tt::tt_metal::DataType::UINT8 ||
        input.dtype() == tt::tt_metal::DataType::UINT32 || input.dtype() == tt::tt_metal::DataType::INT32;
    const bool bfp8_pack_precise = false;  // no typecast in this fork

    return ttnn::device_operation::launch<OperationType>(
        ru::RoutedUnaryParams{
            {op},
            compute_kernel_config,
            resolved_output_mem_cfg,
            out_dtype,
            fp32_dest_acc_en,
            preserve_fp32_precision,
            bfp8_pack_precise,
            local_expert_idx,
            curr_expert_iter,
            expert_iter_length},
        ru::RoutedUnaryInputs{input, global_expert_idx_table, expert_token_counts, std::move(optional_output_tensor)});
}

}  // namespace ttnn::prim
