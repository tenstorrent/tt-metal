// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_matmul_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::device {

RoutedMatmulDeviceOperation::program_factory_t RoutedMatmulDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return RoutedMatmulMcast2DProgramFactory{};
}

void RoutedMatmulDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    TT_FATAL(tensor_args.a.storage_type() == StorageType::DEVICE, "routed_matmul: a must be on device");
    TT_FATAL(tensor_args.b.storage_type() == StorageType::DEVICE, "routed_matmul: b must be on device");

    TT_FATAL(
        tensor_args.global_expert_idx_table.storage_type() == StorageType::DEVICE,
        "routed_matmul: global_expert_idx_table must be on device");
    TT_FATAL(
        tensor_args.global_expert_idx_table.buffer() != nullptr &&
            tensor_args.global_expert_idx_table.buffer()->buffer_type() == BufferType::DRAM,
        "routed_matmul: global_expert_idx_table must be in DRAM");

    TT_FATAL(
        tensor_args.expert_token_counts.storage_type() == StorageType::DEVICE,
        "routed_matmul: expert_token_counts must be on device");
    TT_FATAL(
        tensor_args.expert_token_counts.buffer() != nullptr &&
            tensor_args.expert_token_counts.buffer()->buffer_type() == BufferType::DRAM,
        "routed_matmul: expert_token_counts must be in DRAM");
}

void RoutedMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(operation_attributes, tensor_args);

    // The forked 2D mcast factory only supports the MatmulMultiCoreReuseMultiCastProgramConfig variant
    // (this is the BH-optimized path). Bail loudly for any other config.
    TT_FATAL(
        std::holds_alternative<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>(
            operation_attributes.program_config),
        "routed_matmul: only MatmulMultiCoreReuseMultiCastProgramConfig is supported");
}

RoutedMatmulDeviceOperation::spec_return_value_t RoutedMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.a;
    const auto& b = tensor_args.b;

    const auto& a_shape = a.logical_shape();
    const auto& b_shape = b.logical_shape();

    // Output shape: replace last dim of a with last dim of b.
    std::vector<uint32_t> out_shape_vec(a_shape.cbegin(), a_shape.cend());
    out_shape_vec[out_shape_vec.size() - 1] = b_shape[-1];
    const ttnn::Shape out_shape(out_shape_vec);

    const auto out_dtype = operation_attributes.output_dtype;
    return TensorSpec(
        out_shape,
        tt::tt_metal::TensorLayout(
            out_dtype, tt::tt_metal::PageConfig(Layout::TILE), operation_attributes.output_memory_config));
}

RoutedMatmulDeviceOperation::tensor_return_value_t RoutedMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.a.device());
}

ttsl::hash::hash_t RoutedMatmulDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    // Exclude the three runtime scalars (local_expert_idx, curr_expert_iter,
    // expert_iter_length). Include everything else that can affect kernel
    // compilation / program layout: program config, compute config, memory config,
    // output dtype, and the tensor specs (shape/dtype/layout) of a, b, and the two
    // guard tables, plus whether an optional output was supplied.
    return ttsl::hash::hash_objects_with_default_seed(
        attrs.program_config,
        attrs.compute_kernel_config,
        attrs.output_memory_config,
        attrs.output_dtype,
        args.a.tensor_spec(),
        args.b.tensor_spec(),
        args.global_expert_idx_table.tensor_spec(),
        args.expert_token_counts.tensor_spec(),
        args.optional_output_tensor.has_value());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::device

namespace ttnn::prim {

ttnn::Tensor routed_matmul(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    const ttnn::Tensor& global_expert_idx_table,
    const ttnn::Tensor& expert_token_counts,
    uint32_t local_expert_idx,
    uint32_t curr_expert_iter,
    uint32_t expert_iter_length,
    const ttnn::operations::matmul::MatmulProgramConfig& program_config,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config,
    std::optional<ttnn::Tensor> optional_output_tensor,
    const std::optional<tt::tt_metal::DataType>& output_dtype) {
    namespace rmm = ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::device;
    using OperationType = rmm::RoutedMatmulDeviceOperation;

    const tt::tt_metal::DataType out_dtype = output_dtype.value_or(a.dtype());

    // Resolve output_memory_config: explicit -> caller's value; else inherit from
    // optional_output_tensor (device op will use that tensor anyway); else
    // default to DRAM interleaved.
    const tt::tt_metal::MemoryConfig resolved_output_mem_cfg =
        output_memory_config.has_value()
            ? output_memory_config.value()
            : (optional_output_tensor.has_value()
                   ? optional_output_tensor->memory_config()
                   : tt::tt_metal::MemoryConfig{
                         tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM});

    return ttnn::device_operation::launch<OperationType>(
        rmm::RoutedMatmulParams{
            program_config,
            compute_kernel_config,
            resolved_output_mem_cfg,
            out_dtype,
            local_expert_idx,
            curr_expert_iter,
            expert_iter_length},
        rmm::RoutedMatmulInputs{a, b, global_expert_idx_table, expert_token_counts, std::move(optional_output_tensor)});
}

}  // namespace ttnn::prim
