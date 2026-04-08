// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gdn_fused_device_operation.hpp"
#include "gdn_fused_device_operation_types.hpp"
#include "gdn_fused_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

GdnFusedDeviceOperation::program_factory_t GdnFusedDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // Always return the single-device program factory.
    // The framework's MeshDeviceOperationAdapter automatically adapts it for multi-device
    // via MeshWorkloadFactoryAdapter, creating one program per device.
    return GdnFusedProgramFactory{};
}

void GdnFusedDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(tensor_args.conv_out.storage_type() == StorageType::DEVICE, "conv_out must be on device");
    TT_FATAL(tensor_args.state.storage_type() == StorageType::DEVICE, "state must be on device");
    TT_FATAL(tensor_args.output.storage_type() == StorageType::DEVICE, "output must be on device");
    TT_FATAL(operation_attributes.num_pairs_total > 0, "num_pairs_total must be > 0");
    TT_FATAL(operation_attributes.num_cores > 0, "num_cores must be > 0");
}

GdnFusedDeviceOperation::spec_return_value_t GdnFusedDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // In-place operation: output is pre-allocated
    return tensor_args.output.tensor_spec();
}

GdnFusedDeviceOperation::tensor_return_value_t GdnFusedDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // In-place operation: return the pre-allocated output tensor
    return tensor_args.output;
}

ttsl::hash::hash_t GdnFusedDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto program_factory = select_program_factory(args, tensor_args);

    // Hash program-structure parameters only (NOT buffer addresses which change per call).
    // Include: num_pairs_total, num_cores, state_in_l1, state_is_sharded,
    //          Nv_TP, Nk_TP, repeat_factor, key_dim_tp, and tensor specs.
    return operation::hash_operation<GdnFusedDeviceOperation>(
        args.num_pairs_total,
        args.num_cores,
        args.state_in_l1,
        args.state_is_sharded,
        args.Nv_TP,
        args.Nk_TP,
        args.repeat_factor,
        args.key_dim_tp,
        tensor_args.conv_out.tensor_spec(),
        tensor_args.state.tensor_spec(),
        tensor_args.output.tensor_spec(),
        program_factory.index());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::GdnFusedDeviceOperation::tensor_return_value_t gdn_fused(
    const Tensor& conv_out,
    const Tensor& a_fused,
    const Tensor& b_fused,
    const Tensor& neg_exp_A,
    const Tensor& dt_bias,
    const Tensor& norm_w,
    const Tensor& scale_tt,
    const Tensor& rms_scale_tt,
    const Tensor& rms_eps_tt,
    const Tensor& state,
    const Tensor& output,
    uint32_t num_pairs,
    uint32_t num_cores,
    uint32_t Nv_TP,
    uint32_t Nk_TP,
    uint32_t repeat_factor,
    uint32_t key_dim_tp) {
    using OperationType = ttnn::experimental::prim::GdnFusedDeviceOperation;

    bool state_in_l1 = state.memory_config().buffer_type() == tt::tt_metal::BufferType::L1;
    bool state_is_sharded = state.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED && state_in_l1;

    auto operation_attributes = OperationType::operation_attributes_t{
        .num_pairs_total = num_pairs,
        .num_cores = num_cores,
        .state_in_l1 = state_in_l1,
        .state_is_sharded = state_is_sharded,
        .Nv_TP = Nv_TP,
        .Nk_TP = Nk_TP,
        .repeat_factor = repeat_factor,
        .key_dim_tp = key_dim_tp};

    auto tensor_args = OperationType::tensor_args_t{
        .conv_out = conv_out,
        .a_fused = a_fused,
        .b_fused = b_fused,
        .neg_exp_A = neg_exp_A,
        .dt_bias = dt_bias,
        .norm_w = norm_w,
        .scale_tt = scale_tt,
        .rms_scale_tt = rms_scale_tt,
        .rms_eps_tt = rms_eps_tt,
        .state = state,
        .output = output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
