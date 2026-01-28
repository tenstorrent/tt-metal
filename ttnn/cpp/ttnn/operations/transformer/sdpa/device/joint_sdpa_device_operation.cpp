// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/joint_sdpa_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device.hpp"

#include "ttnn/operations/transformer/sdpa/device/joint_sdpa_device_operation_types.hpp"
#include "ttnn/operations/transformer/sdpa/device/joint_sdpa_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

JointSDPADeviceOperation::program_factory_t JointSDPADeviceOperation::select_program_factory(
    const JointSDPAParams& /*args*/, const JointSDPAInputs& /*tensor_args*/) {
    return JointSDPAProgramFactory{};
}

void JointSDPADeviceOperation::validate_on_program_cache_hit(
    const JointSDPAParams& args, const JointSDPAInputs& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void JointSDPADeviceOperation::validate_on_program_cache_miss(
    const JointSDPAParams& args, const JointSDPAInputs& tensor_args) {
    const auto& input_tensor_q = tensor_args.input_q;
    const auto& input_tensor_k = tensor_args.input_k;
    const auto& input_tensor_v = tensor_args.input_v;
    const auto& joint_tensor_q = tensor_args.joint_q;
    const auto& joint_tensor_k = tensor_args.joint_k;
    const auto& joint_tensor_v = tensor_args.joint_v;

    const std::vector<Tensor> input_tensors = {
        input_tensor_q, input_tensor_k, input_tensor_v, joint_tensor_q, joint_tensor_k, joint_tensor_v};

    // Validate joint strategy is 'rear'
    TT_FATAL(args.joint_strategy == "rear", "Joint strategy must be 'rear'. Got: {}", args.joint_strategy);

    // Validate all tensors have the same dtype
    const auto dtype = input_tensor_q.dtype();
    for (const auto& tensor : input_tensors) {
        TT_FATAL(
            tensor.dtype() == dtype,
            "All tensors must have the same dtype. Expected {}, got {}",
            dtype,
            tensor.dtype());
    }

    // Get shapes
    const auto& q_shape = input_tensor_q.logical_shape();
    const auto& k_shape = input_tensor_k.logical_shape();
    const auto& v_shape = input_tensor_v.logical_shape();
    const auto& joint_q_shape = joint_tensor_q.logical_shape();
    const auto& joint_k_shape = joint_tensor_k.logical_shape();
    const auto& joint_v_shape = joint_tensor_v.logical_shape();

    // Validate storage types and buffers
    for (const auto& tensor : input_tensors) {
        TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "Operands to Joint SDPA need to be on device");
        TT_FATAL(tensor.buffer() != nullptr, "Operands to Joint SDPA need to be allocated in buffers on device");
        TT_FATAL(tensor.layout() == Layout::TILE, "Inputs to Joint SDPA must be tilized");
        TT_FATAL(
            tensor.dtype() == DataType::BFLOAT16 || tensor.dtype() == DataType::BFLOAT8_B,
            "Inputs to Joint SDPA must be BF16 or BF8");
        TT_FATAL(
            tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Operands to Joint SDPA need to be in DRAM");
    }

    // Validate input shapes match
    TT_FATAL(
        k_shape[0] == q_shape[0] && v_shape[0] == q_shape[0],
        "Batch sizes must match. Got Q: {}, K: {}, V: {}",
        q_shape[0],
        k_shape[0],
        v_shape[0]);

    // Validate joint input shapes match
    TT_FATAL(
        joint_k_shape[0] == joint_q_shape[0] && joint_v_shape[0] == joint_q_shape[0],
        "Joint batch sizes must match. Got Q: {}, K: {}, V: {}",
        joint_q_shape[0],
        joint_k_shape[0],
        joint_v_shape[0]);

    // Validate Q and joint Q have same batch size and num heads
    TT_FATAL(
        q_shape[0] == joint_q_shape[0],
        "Q and joint Q must have same batch size. Got Q: {}, joint Q: {}",
        q_shape[0],
        joint_q_shape[0]);

    // Validate head dimensions match
    TT_FATAL(
        k_shape[3] == q_shape[3] && v_shape[3] == q_shape[3],
        "Head dimensions must match. Got Q: {}, K: {}, V: {}",
        q_shape[3],
        k_shape[3],
        v_shape[3]);

    TT_FATAL(
        joint_k_shape[3] == joint_q_shape[3] && joint_v_shape[3] == joint_q_shape[3],
        "Joint head dimensions must match. Got Q: {}, K: {}, V: {}",
        joint_q_shape[3],
        joint_k_shape[3],
        joint_v_shape[3]);

    TT_FATAL(
        q_shape[3] == joint_q_shape[3],
        "Q and joint Q must have same head dimension. Got Q: {}, joint Q: {}",
        q_shape[3],
        joint_q_shape[3]);

    // Validate num_heads relationship
    const auto nqh = q_shape[1];
    const auto nkv = k_shape[1];
    const auto joint_nqh = joint_q_shape[1];
    const auto joint_nkv = joint_k_shape[1];

    TT_FATAL(nqh == nkv, "Q num_heads must be equal to K num_heads. Got Q: {}, K: {}", nqh, nkv);

    TT_FATAL(
        joint_nqh == joint_nkv,
        "Joint Q num_heads must be equal to Joint K num_heads. Got Q: {}, K: {}",
        joint_nqh,
        joint_nkv);
    TT_FATAL(
        joint_nkv == nkv, "Joint K num_heads must be equal to K num_heads. Got Joint K: {}, K: {}", joint_nkv, nkv);

    // Validate chunk sizes if program config is provided
    auto q_chunk_size = args.get_q_chunk_size();
    auto k_chunk_size = args.get_k_chunk_size();

    TT_FATAL(
        q_chunk_size % tt::constants::TILE_WIDTH == 0,
        "q_chunk_size must be divisible by TILE_SIZE. Got q_chunk_size: {}, TILE_SIZE: {}",
        q_chunk_size,
        tt::constants::TILE_WIDTH);
    TT_FATAL(
        k_chunk_size % tt::constants::TILE_WIDTH == 0,
        "k_chunk_size must be divisible by TILE_SIZE. Got k_chunk_size: {}, TILE_SIZE: {}",
        k_chunk_size,
        tt::constants::TILE_WIDTH);

    // Validate padding: Only the sequence dimension may be padded
    auto validate_padding = [](const Tensor& tensor) {
        auto logical_shape = tensor.logical_shape();
        auto padded_shape = tensor.padded_shape();
        TT_FATAL(logical_shape[0] == padded_shape[0], "Padding is not supported on the batch dimension");
        TT_FATAL(logical_shape[1] == padded_shape[1], "Padding is not supported on the num_heads dimension");
        TT_FATAL(logical_shape[3] == padded_shape[3], "Padding is not supported on the head_dim dimension");
    };

    for (const auto& tensor : input_tensors) {
        validate_padding(tensor);
    }
}

JointSDPAResultSpec JointSDPADeviceOperation::compute_output_specs(
    const JointSDPAParams& args, const JointSDPAInputs& tensor_args) {
    const auto& input = tensor_args.input_q;
    const auto& joint_input = tensor_args.joint_q;
    return {
        .output = TensorSpec(
            input.logical_shape(), TensorLayout(input.dtype(), PageConfig(Layout::TILE), args.output_memory_config)),
        .joint_output = TensorSpec(
            joint_input.logical_shape(),
            TensorLayout(joint_input.dtype(), PageConfig(Layout::TILE), args.output_memory_config))};
}

JointSDPAResult JointSDPADeviceOperation::create_output_tensors(
    const JointSDPAParams& args, const JointSDPAInputs& tensor_args) {
    auto output_specs = compute_output_specs(args, tensor_args);
    return {
        .output = create_device_tensor(output_specs.output, tensor_args.input_q.device()),
        .joint_output = create_device_tensor(output_specs.joint_output, tensor_args.joint_q.device())};
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
    const std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    using OperationType = ttnn::prim::JointSDPADeviceOperation;

    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    auto scale_val = scale.value_or(1.0f / std::sqrt(static_cast<float>(input_tensor_q.logical_shape()[-1])));

    auto operation_attributes = OperationType::operation_attributes_t{
        joint_strategy, scale_val, tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG, program_config, kernel_config_val};

    auto tensor_args = OperationType::tensor_args_t{
        .input_q = input_tensor_q,
        .input_k = input_tensor_k,
        .input_v = input_tensor_v,
        .joint_q = joint_tensor_q,
        .joint_k = joint_tensor_k,
        .joint_v = joint_tensor_v};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
