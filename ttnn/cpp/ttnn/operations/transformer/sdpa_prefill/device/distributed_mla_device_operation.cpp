// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "distributed_mla_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::transformer::sdpa_prefill {

DistributedMLADeviceOperation::program_factory_t DistributedMLADeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return DistributedMlaMeshWorkloadFactory{};
}

void DistributedMLADeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(operation_attributes, tensor_args);

    const auto& q_tensor = tensor_args.q;
    const auto& k_tensor = tensor_args.k;
    const auto& v_tensor = tensor_args.v;

    // Q tensor validations
    TT_FATAL(q_tensor.storage_type() == StorageType::DEVICE, "Q tensor must be on device!");
    TT_FATAL(q_tensor.buffer() != nullptr, "Q tensor must be allocated in buffer on device!");
    TT_FATAL(q_tensor.logical_shape().rank() == 4, "Q tensor must have rank 4 [B, NH, S, DH]");

    // K tensor validations
    TT_FATAL(k_tensor.storage_type() == StorageType::DEVICE, "K tensor must be on device!");
    TT_FATAL(k_tensor.buffer() != nullptr, "K tensor must be allocated in buffer on device!");
    TT_FATAL(k_tensor.logical_shape().rank() == 4, "K tensor must have rank 4 [B, NH, S, DH]");

    // V tensor validations
    TT_FATAL(v_tensor.storage_type() == StorageType::DEVICE, "V tensor must be on device!");
    TT_FATAL(v_tensor.buffer() != nullptr, "V tensor must be allocated in buffer on device!");
    TT_FATAL(v_tensor.logical_shape().rank() == 4, "V tensor must have rank 4 [B, NH, S, DH]");

    // Shape compatibility validations
    const auto& q_shape = q_tensor.logical_shape();
    const auto& k_shape = k_tensor.logical_shape();
    const auto& v_shape = v_tensor.logical_shape();
    TT_FATAL(q_shape[0] == k_shape[0] && q_shape[0] == v_shape[0], "Q, K, and V must have same batch size");
    TT_FATAL(q_shape[3] == k_shape[3], "Q and K must have same head dimension");
    TT_FATAL(k_shape[2] == v_shape[2], "K and V must have same sequence length");

    // Page table validation (if provided)
    if (tensor_args.page_table.has_value()) {
        const auto& page_table = tensor_args.page_table.value();
        TT_FATAL(page_table.storage_type() == StorageType::DEVICE, "Page table must be on device!");
        TT_FATAL(page_table.buffer() != nullptr, "Page table must be allocated in buffer on device!");
    }

    // Chunk start idx tensor validation (if provided)
    if (tensor_args.chunk_start_idx_tensor.has_value()) {
        const auto& chunk_start_idx_tensor = tensor_args.chunk_start_idx_tensor.value();
        TT_FATAL(
            chunk_start_idx_tensor.storage_type() == StorageType::DEVICE, "Chunk start idx tensor must be on device!");
        TT_FATAL(
            chunk_start_idx_tensor.buffer() != nullptr,
            "Chunk start idx tensor must be allocated in buffer on device!");
    }
}

void DistributedMLADeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // For now, no additional validation on cache hit
}

DistributedMLADeviceOperation::spec_return_value_t DistributedMLADeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& q_tensor = tensor_args.q;
    auto output_shape = q_tensor.logical_shape();

    // For MLA: output head dimension might be different from input
    if (operation_attributes.head_dim_v.has_value()) {
        output_shape[-1] = operation_attributes.head_dim_v.value();
    }

    auto mem_config = operation_attributes.output_mem_config;
    return TensorSpec(
        output_shape, tt::tt_metal::TensorLayout(q_tensor.dtype(), q_tensor.tensor_spec().page_config(), mem_config));
}

DistributedMLADeviceOperation::tensor_return_value_t DistributedMLADeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_specs, tensor_args.q.device());
}

}  // namespace ttnn::operations::transformer::sdpa_prefill
