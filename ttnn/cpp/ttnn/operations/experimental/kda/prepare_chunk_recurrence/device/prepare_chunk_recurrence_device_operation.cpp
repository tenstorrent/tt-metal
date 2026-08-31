// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "prepare_chunk_recurrence_device_operation.hpp"

#include <array>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"
#include "ttnn/operations/experimental/kda/kda_performance_model.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

PrepareChunkRecurrenceOperation::program_factory_t PrepareChunkRecurrenceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return PrepareChunkRecurrenceProgramFactory{};
}

void PrepareChunkRecurrenceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace kda_factory_detail;
    constexpr std::string_view operation_name = "prepare_chunk_recurrence";
    const auto arch = tt::tt_metal::hal::get_arch();
    TT_FATAL(arch == tt::ARCH::BLACKHOLE, "{} is only supported on Blackhole, got {}", operation_name, arch);
    check_allocated_device_tensor(in.q, operation_name, "q");
    check_layout(in.q, Layout::TILE, operation_name, "q");
    check_dtype(in.q, DataType::BFLOAT16, operation_name, "q");
    check_interleaved(in.q, operation_name, "q");
    check_allocated_device_tensor(in.k, operation_name, "k");
    check_layout(in.k, Layout::TILE, operation_name, "k");
    check_dtype(in.k, DataType::BFLOAT16, operation_name, "k");
    check_interleaved(in.k, operation_name, "k");
    check_allocated_device_tensor(in.v, operation_name, "v");
    check_layout(in.v, Layout::TILE, operation_name, "v");
    check_dtype(in.v, DataType::BFLOAT16, operation_name, "v");
    check_interleaved(in.v, operation_name, "v");
    check_allocated_device_tensor(in.g, operation_name, "g");
    check_layout(in.g, Layout::TILE, operation_name, "g");
    check_dtype(in.g, DataType::BFLOAT16, operation_name, "g");
    check_interleaved(in.g, operation_name, "g");
    check_allocated_device_tensor(in.beta, operation_name, "beta");
    check_layout(in.beta, Layout::TILE, operation_name, "beta");
    check_dtype(in.beta, DataType::FLOAT32, operation_name, "beta");
    check_interleaved(in.beta, operation_name, "beta");
    check_same_device(in.q, in.k, operation_name, "k");
    check_same_device(in.q, in.v, operation_name, "v");
    check_same_device(in.q, in.g, operation_name, "g");
    check_same_device(in.q, in.beta, operation_name, "beta");
    check_output_interleaved(attrs.output_mem_config, operation_name);
    check_compute_config(attrs.compute_kernel_config, operation_name);

    const auto& q_shape = in.q.logical_shape();
    const auto& k_shape = in.k.logical_shape();
    const auto& v_shape = in.v.logical_shape();
    const auto& g_shape = in.g.logical_shape();
    const auto& beta_shape = in.beta.logical_shape();
    TT_FATAL(
        q_shape.rank() == 3 && k_shape.rank() == 3 && v_shape.rank() == 3 && g_shape.rank() == 3,
        "prepare_chunk_recurrence: q, k, v, and g must be rank 3 production-flat tensors");
    TT_FATAL(
        q_shape[0] == 1 && k_shape[0] == 1 && v_shape[0] == 1 && g_shape[0] == 1,
        "prepare_chunk_recurrence: q, k, v, and g must have leading dimension 1");
    TT_FATAL(
        k_shape == q_shape && g_shape == q_shape, "prepare_chunk_recurrence: q, k, and g must have matching shapes");
    TT_FATAL(v_shape[1] == q_shape[1], "prepare_chunk_recurrence: q, k, v, and g must have matching sequence lengths");
    TT_FATAL(
        q_shape[1] > 0 && q_shape[1] % tt::constants::TILE_HEIGHT == 0,
        "prepare_chunk_recurrence: sequence length must be positive and divisible by 32");
    TT_FATAL(attrs.num_heads > 0, "prepare_chunk_recurrence: num_heads must be positive");
    TT_FATAL(
        q_shape[2] % attrs.num_heads == 0 && v_shape[2] % attrs.num_heads == 0,
        "prepare_chunk_recurrence: flat widths must be divisible by num_heads");
    TT_FATAL(
        attrs.key_dim > 0 && attrs.value_dim > 0 && attrs.key_dim % tt::constants::TILE_WIDTH == 0 &&
            attrs.value_dim % tt::constants::TILE_WIDTH == 0,
        "prepare_chunk_recurrence: K and V must be positive and tile aligned");
    TT_FATAL(
        q_shape[2] == attrs.num_heads * attrs.key_dim && v_shape[2] == attrs.num_heads * attrs.value_dim &&
            q_shape[1] == attrs.num_chunks * tt::constants::TILE_HEIGHT,
        "prepare_chunk_recurrence: flat input shapes must match operation attributes");

    TT_FATAL(
        beta_shape.rank() == 4 && beta_shape[0] == attrs.num_heads && beta_shape[1] == attrs.num_chunks &&
            beta_shape[2] == tt::constants::TILE_HEIGHT && beta_shape[3] == 1,
        "prepare_chunk_recurrence: beta shape must be [num_heads, num_chunks, 32, 1]");
    constexpr uint32_t allowed_bf16_mask = 0x37;
    TT_FATAL(
        (attrs.output_bf16_mask & ~allowed_bf16_mask) == 0,
        "unsupported KDA prep BF16 mask 0x{:x}",
        attrs.output_bf16_mask);
}

PrepareChunkRecurrenceOperation::spec_return_value_t PrepareChunkRecurrenceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    const auto spec = [&](const Shape& shape, uint32_t index) {
        const auto dtype = (attrs.output_bf16_mask & (1U << index)) ? DataType::BFLOAT16 : DataType::FLOAT32;
        return TensorSpec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), attrs.output_mem_config));
    };
    const auto BH = attrs.num_heads;
    const auto NC = attrs.num_chunks;
    constexpr uint32_t C = tt::constants::TILE_HEIGHT;
    const auto K = attrs.key_dim;
    const auto V = attrs.value_dim;
    return {
        spec(Shape({BH, NC, C, V}), 0),
        spec(Shape({BH, NC, C, K}), 1),
        spec(Shape({BH, NC, C, K}), 2),
        spec(Shape({BH, NC, C, C}), 3),
        spec(Shape({BH, NC, K, C}), 4),
        spec(Shape({BH, NC, K, 1}), 5),
        spec(Shape({BH, NC, C, C}), 6)};
}

PrepareChunkRecurrenceOperation::tensor_return_value_t PrepareChunkRecurrenceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(attrs, in)) {
        outputs.push_back(create_device_tensor(spec, in.q.device()));
    }
    return outputs;
}

tt::tt_metal::operation::OpPerformanceModelGeneral<PrepareChunkRecurrenceOperation::tensor_return_value_t>
PrepareChunkRecurrenceOperation::create_op_performance_model(
    const operation_attributes_t& attrs, const tensor_args_t& in, tensor_return_value_t& outputs) {
    using namespace kda_performance_model;

    const auto work = prepare_chunk_recurrence_work(attrs.num_heads, attrs.num_chunks, attrs.key_dim, attrs.value_dim);
    const std::array<const Tensor*, 5> inputs = {&in.q, &in.k, &in.v, &in.g, &in.beta};
    return make_profiler_model(work, inputs, outputs, attrs.compute_kernel_config.math_fidelity);
}

std::vector<Tensor> prepare_chunk_recurrence(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    uint32_t num_heads,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    uint32_t output_bf16_mask) {
    const auto& q_shape = q.logical_shape();
    const auto& v_shape = v.logical_shape();
    TT_FATAL(
        q_shape.rank() == 3 && v_shape.rank() == 3,
        "prepare_chunk_recurrence: q and v must be rank 3 production-flat tensors");
    TT_FATAL(num_heads > 0, "prepare_chunk_recurrence: num_heads must be positive");
    TT_FATAL(
        q_shape[1] > 0 && q_shape[1] % tt::constants::TILE_HEIGHT == 0,
        "prepare_chunk_recurrence: sequence length must be positive and divisible by 32");
    TT_FATAL(
        q_shape[2] % num_heads == 0 && v_shape[2] % num_heads == 0,
        "prepare_chunk_recurrence: flat widths must be divisible by num_heads");
    const uint32_t num_chunks = q_shape[1] / tt::constants::TILE_HEIGHT;
    const uint32_t key_dim = q_shape[2] / num_heads;
    const uint32_t value_dim = v_shape[2] / num_heads;
    return ttnn::device_operation::launch<PrepareChunkRecurrenceOperation>(
        PrepareChunkRecurrenceParams{
            .num_heads = num_heads,
            .num_chunks = num_chunks,
            .key_dim = key_dim,
            .value_dim = value_dim,
            .output_bf16_mask = output_bf16_mask,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config},
        PrepareChunkRecurrenceInputs{.q = q, .k = k, .v = v, .g = g, .beta = beta});
}

}  // namespace ttnn::experimental::prim
