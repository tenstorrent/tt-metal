// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/recipe/sdpa_recipe_joint.hpp"
#include "ttnn/operations/transformer/sdpa/device/joint_sdpa_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include <cmath>

#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operation.hpp"

#include "ttnn/operations/transformer/sdpa/device/joint_sdpa_device_operation_types.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_perf_model.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {
void JointSDPADeviceOperation::validate_on_program_cache_miss(
    const JointSDPAParams& args, const JointSDPAInputs& tensor_args) {
    // resolve_joint_recipe resolves the recipe before the program-cache lookup; recipe_program checks shapes, types
    // and layouts when it builds the program.
    TT_FATAL(args.joint_strategy == "rear", "Joint strategy must be 'rear'. Got: {}", args.joint_strategy);
    TT_FATAL(args.precision.has_value(), "Joint SDPA: the precision recipe is resolved before dispatch");
    (void)tensor_args;
}

JointSDPAResultSpec JointSDPADeviceOperation::compute_output_specs(
    const JointSDPAParams& args, const JointSDPAInputs& tensor_args) {
    return ttnn::operations::transformer::sdpa::detail::joint_recipe_output_specs(args, tensor_args);
}

JointSDPAResult JointSDPADeviceOperation::create_output_tensors(
    const JointSDPAParams& args, const JointSDPAInputs& tensor_args) {
    auto output_specs = compute_output_specs(args, tensor_args);
    return {
        create_device_tensor(output_specs[JOINT_SDPA_OUTPUT_IDX], tensor_args.input_q.device()),
        create_device_tensor(output_specs[JOINT_SDPA_JOINT_OUTPUT_IDX], tensor_args.joint_q.device())};
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensors> JointSDPADeviceOperation::create_op_performance_model(
    const JointSDPAParams& args, const JointSDPAInputs& tensor_args, JointSDPAResult& output_tensors) {
    Tensors input_tensors = {
        tensor_args.input_q,
        tensor_args.input_k,
        tensor_args.input_v,
        tensor_args.joint_q,
        tensor_args.joint_k,
        tensor_args.joint_v};

    auto& output_tensor = output_tensors[JOINT_SDPA_OUTPUT_IDX];
    auto arch = output_tensor.storage_type() == StorageType::DEVICE ? output_tensor.device()->arch()
                                                                    : ttnn::GetDefaultDevice()->arch();

    if (arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
        log_warning(tt::LogOp, "JointSDPA perf model does not support arch '{}'", enchantum::to_string(arch));
        return operation::OpPerformanceModelGeneral<Tensors>(input_tensors, output_tensors, 0);
    }

    const auto& q_shape = tensor_args.input_q.logical_shape();
    const auto& k_shape = tensor_args.input_k.logical_shape();
    const auto& v_shape = tensor_args.input_v.logical_shape();
    const auto& joint_q_shape = tensor_args.joint_q.logical_shape();

    CoreCoord grid = args.program_config.has_value() ? args.program_config->compute_with_storage_grid_size
                                                     : output_tensor.device()->compute_with_storage_grid_size();
    tt::tt_metal::MathFidelity fidelity = ttnn::operations::transformer::sdpa::detail::recipe_math_fidelity(args.precision, tensor_args.input_k.dtype());

    const uint32_t B = q_shape[0];
    const uint32_t NQH = q_shape[1];
    const uint32_t Sq = q_shape[2];
    const uint32_t Sk = k_shape[2];
    const uint32_t Sj = joint_q_shape[2];
    const uint32_t DH = q_shape[3];
    const uint32_t DV = v_shape[3];

    // JointSDPA operates on concatenated sequences (validated in program factory)
    // cat_Sq = Sq + Sj, cat_Sk = Sk + Sj
    const uint32_t cat_Sq = Sq + Sj;
    const uint32_t cat_Sk = Sk + Sj;

    // Single attention pass over concatenated dimensions, non-causal
    int ideal_cycles = operations::transformer::sdpa::compute_sdpa_ideal_cycles(
        B, NQH, cat_Sq, cat_Sk, DH, DV, false, fidelity, grid.x * grid.y);

    return operation::OpPerformanceModelGeneral<Tensors>(input_tensors, output_tensors, ideal_cycles);
}

}  // namespace ttnn::prim

namespace ttnn::prim {

JointSDPAResult joint_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& joint_tensor_q,
    const ttnn::Tensor& joint_tensor_k,
    const ttnn::Tensor& joint_tensor_v,
    const std::string& joint_strategy,
    const std::optional<ttnn::operations::transformer::SDPAProgramConfig>& program_config,
    const std::optional<float> scale,
    std::optional<ttnn::operations::transformer::SDPAPrecision> precision) {
    using OperationType = ttnn::prim::JointSDPADeviceOperation;

    auto scale_val = scale.value_or(1.0f / std::sqrt(static_cast<float>(input_tensor_q.logical_shape()[-1])));

    auto operation_attributes = OperationType::operation_attributes_t{
        joint_strategy, scale_val, tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG, program_config};
    operation_attributes.precision = precision.value_or(ttnn::operations::transformer::SDPAPrecision::STANDARD);

    auto tensor_args = OperationType::tensor_args_t{
        .input_q = input_tensor_q,
        .input_k = input_tensor_k,
        .input_v = input_tensor_v,
        .joint_q = joint_tensor_q,
        .joint_k = joint_tensor_k,
        .joint_v = joint_tensor_v};

    ttnn::operations::transformer::sdpa::detail::resolve_joint_recipe(operation_attributes, tensor_args);
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
