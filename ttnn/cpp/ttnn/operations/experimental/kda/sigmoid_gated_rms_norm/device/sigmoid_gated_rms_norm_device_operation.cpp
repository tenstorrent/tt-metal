// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "sigmoid_gated_rms_norm_device_operation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>

#include <tt-metalium/constants.hpp>
#include <tt-logger/tt-logger.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/kda_performance_model.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {
namespace {

void check_tiled_device(const Tensor& tensor, const char* name, std::initializer_list<DataType> dtypes) {
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE && tensor.buffer() != nullptr,
        "sigmoid_gated_rms_norm: {} must be an allocated device tensor",
        name);
    TT_FATAL(tensor.layout() == Layout::TILE, "sigmoid_gated_rms_norm: {} must use TILE layout", name);
    TT_FATAL(!tensor.is_sharded(), "sigmoid_gated_rms_norm: {} must use interleaved memory", name);
    TT_FATAL(
        std::find(dtypes.begin(), dtypes.end(), tensor.dtype()) != dtypes.end(),
        "sigmoid_gated_rms_norm: {} has unsupported dtype {}",
        name,
        tensor.dtype());
}

}  // namespace

SigmoidGatedRmsNormOperation::program_factory_t SigmoidGatedRmsNormOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return SigmoidGatedRmsNormProgramFactory{};
}

void SigmoidGatedRmsNormOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    check_tiled_device(in.input, "input", {DataType::FLOAT32, DataType::BFLOAT16});
    check_tiled_device(in.gate, "gate", {DataType::BFLOAT16});
    check_tiled_device(in.weight, "weight", {DataType::BFLOAT16});
    TT_FATAL(
        in.input.device() == in.gate.device() && in.input.device() == in.weight.device(),
        "sigmoid_gated_rms_norm: input, gate, and weight must be on the same device");
    TT_FATAL(
        attrs.output_dtype == DataType::FLOAT32 || attrs.output_dtype == DataType::BFLOAT16,
        "sigmoid_gated_rms_norm: output_dtype must be FLOAT32 or BFLOAT16");
    TT_FATAL(
        !attrs.output_mem_config.is_sharded(),
        "sigmoid_gated_rms_norm: output memory configuration must be interleaved");
    TT_FATAL(
        std::isfinite(attrs.epsilon) && attrs.epsilon > 0.0F,
        "sigmoid_gated_rms_norm: epsilon must be finite and positive");
    TT_FATAL(
        !attrs.compute_kernel_config.packer_l1_acc,
        "sigmoid_gated_rms_norm: packer_l1_acc=true is unsupported because the compute kernel does not accumulate "
        "through L1");

    const auto& input_shape = in.input.logical_shape();
    const auto& gate_shape = in.gate.logical_shape();
    const auto& weight_shape = in.weight.logical_shape();
    TT_FATAL(input_shape.rank() == 3, "sigmoid_gated_rms_norm: input must be [B*H,T,V]");
    TT_FATAL(gate_shape.rank() == 3, "sigmoid_gated_rms_norm: gate must be [B,T,H*V]");
    TT_FATAL(weight_shape.rank() == 1, "sigmoid_gated_rms_norm: weight must be [V]");
    TT_FATAL(
        input_shape[0] == attrs.batch * attrs.num_heads && input_shape[1] == attrs.sequence &&
            input_shape[2] == attrs.value_dim,
        "sigmoid_gated_rms_norm: input shape does not match derived attributes");
    TT_FATAL(
        gate_shape[0] == attrs.batch && gate_shape[1] == attrs.sequence &&
            gate_shape[2] == attrs.num_heads * attrs.value_dim,
        "sigmoid_gated_rms_norm: gate must have shape [B,T,H*V]");
    TT_FATAL(in.weight.logical_volume() == attrs.value_dim, "sigmoid_gated_rms_norm: weight volume must equal V");
    TT_FATAL(attrs.batch > 0, "sigmoid_gated_rms_norm: batch must be positive");
    TT_FATAL(
        attrs.sequence > 0 && attrs.sequence % tt::constants::TILE_HEIGHT == 0,
        "sigmoid_gated_rms_norm: sequence must be positive and tile aligned");
    TT_FATAL(
        attrs.value_dim > 0 && attrs.value_dim % tt::constants::TILE_WIDTH == 0,
        "sigmoid_gated_rms_norm: value_dim must be positive and tile aligned");
}

SigmoidGatedRmsNormOperation::spec_return_value_t SigmoidGatedRmsNormOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    return {TensorSpec(
        Shape({attrs.batch, attrs.sequence, attrs.num_heads * attrs.value_dim}),
        TensorLayout(attrs.output_dtype, PageConfig(Layout::TILE), attrs.output_mem_config))};
}

SigmoidGatedRmsNormOperation::tensor_return_value_t SigmoidGatedRmsNormOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    return {create_device_tensor(specs[0], in.input.device())};
}
tt::tt_metal::operation::OpPerformanceModelGeneral<SigmoidGatedRmsNormOperation::tensor_return_value_t>
SigmoidGatedRmsNormOperation::create_op_performance_model(
    const operation_attributes_t& attrs, const tensor_args_t& in, tensor_return_value_t& outputs) {
    using namespace kda_performance_model;

    constexpr std::size_t input_count = 3;
    auto fallback = to_profiler_model<tensor_return_value_t>(zero_estimate(input_count, outputs.size()));
    if (in.input.storage_type() != StorageType::DEVICE || !in.input.is_allocated() || in.input.device() == nullptr) {
        log_warning(tt::LogOp, "KDA sigmoid_gated_rms_norm performance model expected an allocated device input");
        return fallback;
    }

    auto* device = in.input.device();
    if (device->arch() != tt::ARCH::BLACKHOLE) {
        log_warning(tt::LogOp, "KDA sigmoid_gated_rms_norm performance model supports Blackhole only");
        return fallback;
    }

    const auto work = sigmoid_gated_rms_norm_work(attrs.batch, attrs.num_heads, attrs.sequence, attrs.value_dim);
    if (!work) {
        return fallback;
    }

    const std::array<const Tensor*, input_count> input_tensors = {&in.input, &in.gate, &in.weight};
    std::array<KdaTensorTraffic, input_count> input_traffic;
    for (std::size_t index = 0; index < input_tensors.size(); ++index) {
        const auto traffic = tensor_traffic(*input_tensors[index]);
        if (!traffic) {
            return fallback;
        }
        input_traffic[index] = *traffic;
    }

    std::vector<KdaTensorTraffic> output_traffic;
    output_traffic.reserve(outputs.size());
    for (const auto& output : outputs) {
        const auto traffic = tensor_traffic(output);
        if (!traffic) {
            return fallback;
        }
        output_traffic.push_back(*traffic);
    }

    const auto grid = device->compute_with_storage_grid_size();
    const auto estimate_result = estimate(
        *work,
        input_traffic,
        output_traffic,
        static_cast<uint64_t>(grid.x) * grid.y,
        device->get_clock_rate_mhz(),
        attrs.compute_kernel_config.math_fidelity);
    return to_profiler_model<tensor_return_value_t>(estimate_result);
}

Tensor sigmoid_gated_rms_norm(
    const Tensor& input,
    const Tensor& gate,
    const Tensor& weight,
    uint32_t num_heads,
    float epsilon,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    DataType output_dtype) {
    const auto& input_shape = input.logical_shape();
    TT_FATAL(input_shape.rank() == 3, "sigmoid_gated_rms_norm: input must be [B*H,T,V]");
    TT_FATAL(num_heads > 0, "sigmoid_gated_rms_norm: num_heads must be positive");
    TT_FATAL(
        input_shape[0] % num_heads == 0, "sigmoid_gated_rms_norm: leading dimension must be divisible by num_heads");
    const uint32_t batch = input_shape[0] / num_heads;
    auto results = ttnn::device_operation::launch<SigmoidGatedRmsNormOperation>(
        SigmoidGatedRmsNormParams{
            .batch = batch,
            .num_heads = num_heads,
            .sequence = static_cast<uint32_t>(input_shape[1]),
            .value_dim = static_cast<uint32_t>(input_shape[2]),
            .epsilon = epsilon,
            .output_mem_config = output_mem_config,
            .output_dtype = output_dtype,
            .compute_kernel_config = compute_kernel_config},
        SigmoidGatedRmsNormInputs{.input = input, .gate = gate, .weight = weight});
    return results[0];
}

}  // namespace ttnn::experimental::prim
