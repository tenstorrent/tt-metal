// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rms_norm_ttnn_device_operation.hpp"

#include <cstdlib>
#include <cstring>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::bringup::rms_norm_ttnn {

void RmsNormDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    TT_FATAL(
        tensor_args.input.storage_type() == ttnn::StorageType::DEVICE,
        "rms_norm_ttnn: input_tensor must be resident on a device");
}

void RmsNormDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {}

RmsNormDeviceOperation::spec_return_value_t RmsNormDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    if (attrs.inplace) {
        return input.tensor_spec();
    }
    // ttnn.allocate_tensor_on_device(Shape(input.shape), input.dtype, input.layout, device, memory_config)
    return tt::tt_metal::TensorSpec(
        input.logical_shape(),
        tt::tt_metal::TensorLayout(input.dtype(), tt::tt_metal::PageConfig(input.layout()), attrs.output_mem_config));
}

RmsNormDeviceOperation::tensor_return_value_t RmsNormDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    if (attrs.inplace) {
        // `inplace`: the output IS the input -- cb_output_tiles aliases its shard.
        return tensor_args.input;
    }
    return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input.device());
}

namespace {
// The builder dedupes the resident-shard L1 charge by buffer identity; aliasing is part of the key.
bool residual_aliases_input(const RmsNormInputs& t) {
    return t.residual.has_value() && t.residual->is_allocated() && t.input.is_allocated() &&
           t.residual->buffer() == t.input.buffer();
}

// The env switches that change the kernels' defines (read by the builder).
std::string env_or_empty(const char* name) {
    const char* v = std::getenv(name);  // diagnostic: perf/measurement knob, same result
    return v == nullptr ? std::string() : std::string(v);
}
}  // namespace

ttsl::hash::hash_t RmsNormDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // Everything create_program_descriptor reads except buffer addresses: the tensors' specs (shape,
    // dtype, layout, memory config incl. shard spec), which optional operands are present, epsilon,
    // the resolved compute config, subblock_w, inplace, the requested output placement, the
    // residual-sum output's spec (`return_residual_sum`: absent when off), and the two
    // kernel-define env switches.  The device (grid, L1 size, core coordinates) is fixed per
    // program cache.
    static const std::string env_key = env_or_empty("RMS_STAGE_ZONES") + "|" + env_or_empty("RMS_ABLATE");
    const auto& cc = attrs.compute_config;
    std::vector<uint32_t> unpack_modes;
    unpack_modes.reserve(cc.unpack_to_dest_mode.size());
    for (auto m : cc.unpack_to_dest_mode) {
        unpack_modes.push_back(static_cast<uint32_t>(m));
    }
    auto spec_of = [](const std::optional<Tensor>& t) -> std::optional<tt::tt_metal::TensorSpec> {
        if (!t.has_value()) {
            return std::nullopt;
        }
        return t->tensor_spec();
    };
    return tt::tt_metal::operation::hash_operation<RmsNormDeviceOperation>(
        attrs.epsilon,
        static_cast<uint32_t>(cc.math_fidelity),
        cc.fp32_dest_acc_en,
        cc.dst_full_sync_en,
        unpack_modes,
        cc.bfp8_pack_precise,
        cc.math_approx_mode,
        cc.enable_trisc2_rvv,
        attrs.subblock_w,
        attrs.inplace,
        attrs.output_mem_config,
        tensor_args.input.tensor_spec(),
        spec_of(tensor_args.weight),
        spec_of(tensor_args.bias),
        spec_of(tensor_args.residual),
        residual_aliases_input(tensor_args),
        spec_of(tensor_args.residual_sum),
        env_key);
}

}  // namespace ttnn::operations::bringup::rms_norm_ttnn

namespace ttnn::prim::bringup {

ttnn::Tensor rms_norm_ttnn(
    const ttnn::Tensor& input,
    const std::optional<ttnn::Tensor>& weight,
    const std::optional<ttnn::Tensor>& bias,
    const std::optional<ttnn::Tensor>& residual,
    double epsilon,
    const tt::tt_metal::ComputeConfigDescriptor& compute_config,
    uint32_t subblock_w,
    bool inplace,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<ttnn::Tensor>& residual_sum) {
    using OperationType = ttnn::operations::bringup::rms_norm_ttnn::RmsNormDeviceOperation;
    auto attrs = OperationType::operation_attributes_t{
        .epsilon = epsilon,
        .compute_config = compute_config,
        .subblock_w = subblock_w,
        .inplace = inplace,
        .output_mem_config = output_mem_config};
    auto args = OperationType::tensor_args_t{
        .input = input, .weight = weight, .bias = bias, .residual = residual, .residual_sum = residual_sum};
    return ttnn::device_operation::launch<OperationType>(attrs, args);
}

}  // namespace ttnn::prim::bringup
