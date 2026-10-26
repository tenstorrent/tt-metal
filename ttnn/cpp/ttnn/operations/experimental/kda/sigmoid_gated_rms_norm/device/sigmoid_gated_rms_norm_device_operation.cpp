// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "sigmoid_gated_rms_norm_device_operation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/kda_performance_model.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {
namespace {

void check_tiled_device(
    const Tensor& tensor, const char* name, std::initializer_list<DataType> dtypes, bool may_be_sharded = false) {
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE && tensor.buffer() != nullptr,
        "sigmoid_gated_rms_norm: {} must be an allocated device tensor",
        name);
    TT_FATAL(tensor.layout() == Layout::TILE, "sigmoid_gated_rms_norm: {} must use TILE layout", name);
    TT_FATAL(!tensor.is_sharded() || may_be_sharded, "sigmoid_gated_rms_norm: {} must use interleaved memory", name);
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
    // The input and the activated gate may be sharded (one item per shard, round-robin, keeps a core's reads local).
    check_tiled_device(in.input, "input", {DataType::FLOAT32, DataType::BFLOAT16}, true);
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
    TT_FATAL(
        input_shape.rank() == 3 || (input_shape.rank() == 4 && input_shape[2] % tt::constants::TILE_HEIGHT == 0),
        "sigmoid_gated_rms_norm: input must be [B*H,T,V] or [B*H,C,L,V] (T = C*L, L tile aligned)");
    TT_FATAL(gate_shape.rank() == 3, "sigmoid_gated_rms_norm: gate must be rank 3 [B,T,W]");
    TT_FATAL(weight_shape.rank() == 1, "sigmoid_gated_rms_norm: weight must be [V]");
    TT_FATAL(
        input_shape[0] == attrs.batch * attrs.num_heads && input_shape[-1] == attrs.value_dim &&
            in.input.logical_volume() == uint64_t{attrs.batch} * attrs.num_heads * attrs.sequence * attrs.value_dim,
        "sigmoid_gated_rms_norm: input shape does not match derived attributes");
    TT_FATAL(
        attrs.gate_column_offset % tt::constants::TILE_WIDTH == 0,
        "sigmoid_gated_rms_norm: gate_column_offset must be tile aligned");
    TT_FATAL(
        attrs.activated_heads <= attrs.num_heads && (attrs.activated_heads == 0) == !in.activated_gate.has_value(),
        "sigmoid_gated_rms_norm: activated_heads {} needs an activated_gate and at most num_heads {}",
        attrs.activated_heads,
        attrs.num_heads);
    if (in.activated_gate.has_value()) {
        const auto& activated = *in.activated_gate;
        // FP32 only: the product reads it where it reads the kernel's own FP32 sigmoid buffer.
        check_tiled_device(activated, "activated_gate", {DataType::FLOAT32}, true);
        TT_FATAL(
            activated.device() == in.input.device(),
            "sigmoid_gated_rms_norm: activated_gate must be on the input's device");
        const auto& activated_shape = activated.logical_shape();
        TT_FATAL(
            activated_shape.rank() == 3 && activated_shape[0] == attrs.batch * attrs.activated_heads &&
                activated_shape[1] == attrs.sequence && activated_shape[2] == attrs.value_dim,
            "sigmoid_gated_rms_norm: activated_gate {} must be [B * activated_heads, T, V], laid out as the input",
            activated_shape);
    }
    const uint64_t gate_width = uint64_t{attrs.num_heads - attrs.activated_heads} * attrs.value_dim;
    TT_FATAL(
        gate_shape[0] == attrs.batch && gate_shape[1] == attrs.sequence &&
            gate_shape[2] >= attrs.gate_column_offset + gate_width,
        "sigmoid_gated_rms_norm: gate {} must be [B, T, W] and hold (H - activated_heads)*V = {} columns at "
        "gate_column_offset {}",
        gate_shape,
        gate_width,
        attrs.gate_column_offset);
    TT_FATAL(in.weight.logical_volume() == attrs.value_dim, "sigmoid_gated_rms_norm: weight volume must equal V");
    TT_FATAL(
        in.prefetch_source.has_value() == in.prefetch_jobs.has_value() &&
            (in.prefetch_jobs.has_value() || attrs.prefetch_cores.empty()),
        "sigmoid_gated_rms_norm: a prefetch needs its source, its job table and its cores");
    if (in.prefetch_jobs.has_value()) {
        const auto& jobs = *in.prefetch_jobs;
        TT_FATAL(
            jobs.storage_type() == StorageType::DEVICE && jobs.buffer() != nullptr &&
                jobs.dtype() == DataType::UINT32 && jobs.layout() == Layout::ROW_MAJOR && !jobs.is_sharded() &&
                jobs.logical_shape()[-1] == 16 && jobs.logical_volume() / 16 >= attrs.prefetch_cores.size(),
            "sigmoid_gated_rms_norm: prefetch_jobs must be an interleaved ROW_MAJOR UINT32 [..., jobs, 16] device "
            "tensor "
            "with a row per prefetch core");
        TT_FATAL(
            in.prefetch_source->storage_type() == StorageType::DEVICE && in.prefetch_source->buffer() != nullptr &&
                !in.prefetch_source->is_sharded(),
            "sigmoid_gated_rms_norm: prefetch_source must be an interleaved device tensor");
    }
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

    const double rows = static_cast<double>(attrs.batch) * attrs.num_heads * attrs.sequence;
    const double elements = rows * attrs.value_dim;
    const KdaFpuWork work{
        .fpu_multiply_ops = 4.0 * elements,
        .fpu_add_ops = rows,
        .fpu_reduction_ops = rows * (attrs.value_dim - 1.0),
    };
    const std::array<const Tensor*, 3> inputs = {&in.input, &in.gate, &in.weight};
    return make_profiler_model(work, inputs, outputs, attrs.compute_kernel_config.math_fidelity);
}

Tensor sigmoid_gated_rms_norm(
    const Tensor& input,
    const Tensor& gate,
    const Tensor& weight,
    uint32_t num_heads,
    float epsilon,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    DataType output_dtype,
    uint32_t gate_column_offset,
    const std::optional<Tensor>& activated_gate,
    const std::optional<Tensor>& prefetch_source,
    const std::optional<Tensor>& prefetch_jobs,
    const std::vector<tt::tt_metal::CoreCoord>& prefetch_cores,
    uint32_t prefetch_l1_begin,
    uint32_t prefetch_l1_end) {
    const auto& input_shape = input.logical_shape();
    TT_FATAL(
        input_shape.rank() == 3 || input_shape.rank() == 4,
        "sigmoid_gated_rms_norm: input must be [B*H,T,V] or [B*H,C,L,V]");
    TT_FATAL(num_heads > 0, "sigmoid_gated_rms_norm: num_heads must be positive");
    TT_FATAL(
        input_shape[0] % num_heads == 0, "sigmoid_gated_rms_norm: leading dimension must be divisible by num_heads");
    const uint32_t batch = input_shape[0] / num_heads;
    auto results = ttnn::device_operation::launch<SigmoidGatedRmsNormOperation>(
        SigmoidGatedRmsNormParams{
            .batch = batch,
            .num_heads = num_heads,
            .sequence = static_cast<uint32_t>(input.logical_volume() / input_shape[0] / input_shape[-1]),
            .value_dim = static_cast<uint32_t>(input_shape[-1]),
            .epsilon = epsilon,
            .output_mem_config = output_mem_config,
            .output_dtype = output_dtype,
            .compute_kernel_config = compute_kernel_config,
            .gate_column_offset = gate_column_offset,
            .activated_heads =
                activated_gate.has_value() ? static_cast<uint32_t>(activated_gate->logical_shape()[0] / batch) : 0u,
            .prefetch_cores = prefetch_cores,
            .prefetch_l1_begin = prefetch_l1_begin,
            .prefetch_l1_end = prefetch_l1_end},
        SigmoidGatedRmsNormInputs{
            .input = input,
            .gate = gate,
            .weight = weight,
            .activated_gate = activated_gate,
            .prefetch_source = prefetch_source,
            .prefetch_jobs = prefetch_jobs});
    return results[0];
}

}  // namespace ttnn::experimental::prim
