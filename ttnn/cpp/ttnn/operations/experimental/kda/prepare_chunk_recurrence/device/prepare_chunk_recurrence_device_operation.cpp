// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "prepare_chunk_recurrence_device_operation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
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
    for (const auto& [tensor, name] : std::array{
             std::pair{&in.q, "q"},
             std::pair{&in.k, "k"},
             std::pair{&in.v, "v"},
             std::pair{&in.g, "g"},
             std::pair{&in.beta, "beta"}}) {
        check_allocated_device_tensor(*tensor, operation_name, name);
        check_layout(*tensor, Layout::TILE, operation_name, name);
        check_dtype(*tensor, DataType::BFLOAT16, operation_name, name);
        check_interleaved(*tensor, operation_name, name);
        check_same_device(in.q, *tensor, operation_name, name);
    }
    const auto* mesh = in.q.device();
    const auto arch = mesh->arch();
    TT_FATAL(arch == tt::ARCH::BLACKHOLE, "{} is only supported on Blackhole, got {}", operation_name, arch);
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

    // Beta logits are num_heads columns, from a tile-aligned offset, of a token-major [1, num_chunks * 32, columns]
    // tensor such as the fused input projection.
    TT_FATAL(
        beta_shape.rank() == 3 && beta_shape[0] == 1 &&
            beta_shape[1] == attrs.num_chunks * tt::constants::TILE_HEIGHT &&
            beta_shape[2] >= uint64_t{attrs.beta_logits_column_offset} + attrs.num_heads &&
            attrs.beta_logits_column_offset % tt::constants::TILE_WIDTH == 0,
        "prepare_chunk_recurrence: beta logits must be token-major [1, num_chunks * 32, columns] with a "
        "tile-aligned column offset and num_heads columns after it");
    constexpr uint32_t allowed_bf16_mask = 0x37;
    TT_FATAL(
        (attrs.output_bf16_mask & ~allowed_bf16_mask) == 0,
        "unsupported KDA prep BF16 mask 0x{:x}",
        attrs.output_bf16_mask);

    // The batched inverse holds all of a core's work items at once, so the heads' chunks must spread over the cores
    // thinly enough for the program's buffers to fit the L1 free per core.
    const auto distribution = kda_factory_detail::distribute_prep(
        mesh->compute_with_storage_grid_size(),
        attrs.num_heads * attrs.num_chunks,
        std::numeric_limits<uint32_t>::max());
    const uint32_t max_items = *std::max_element(distribution.wi_count.begin(), distribution.wi_count.end());
    const uint64_t dfb_bytes = prepare_chunk_recurrence_l1_bytes(attrs, max_items);
    const uint64_t l1_bytes =
        mesh->l1_size_per_core() - mesh->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    TT_FATAL(
        dfb_bytes <= l1_bytes,
        "{}: {} work items per core ({} heads x {} chunks) need {} bytes of L1, {} of them per item for the batched "
        "inverse, only {} are available",
        operation_name,
        max_items,
        attrs.num_heads,
        attrs.num_chunks,
        dfb_bytes,
        prepare_chunk_recurrence_inverse_tiles_per_item * tt::tile_size(tt::DataFormat::Float32),
        l1_bytes);
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

    constexpr double chunk = tt::constants::TILE_HEIGHT;
    constexpr double inverse_flops = chunk * (chunk - 1.0) * (chunk + 1.0) / 3.0;
    const double instances = static_cast<double>(attrs.num_heads) * attrs.num_chunks;
    const double key_dim = attrs.key_dim;
    const double value_dim = attrs.value_dim;
    const KdaFpuWork work{
        .fpu_matrix_flops = instances * (4.0 * chunk * chunk * key_dim + inverse_flops),
        .fpu_multiply_ops = instances * (10.0 * chunk * key_dim + chunk * value_dim),
        .fpu_add_ops = instances * (2.0 * chunk + (chunk - 1.0) * key_dim + chunk * key_dim + chunk * chunk),
        .fpu_reduction_ops = instances * 2.0 * chunk * (key_dim - 1.0),
    };
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
    uint32_t output_bf16_mask,
    const std::optional<Tensor>& actual_start,
    const std::optional<Tensor>& actual_end,
    uint32_t sequence_parallel_axis,
    float gate_scale,
    uint32_t beta_logits_column_offset) {
    TT_FATAL(std::isfinite(gate_scale), "prepare_chunk_recurrence: gate_scale must be finite");
    TT_FATAL(!actual_end || actual_start, "prepare_chunk_recurrence: actual_end requires actual_start");
    if (actual_start) {
        kda_factory_detail::check_actual_start(q, *actual_start, "prepare_chunk_recurrence");
    }
    if (actual_end) {
        kda_factory_detail::check_actual_start(q, *actual_end, "prepare_chunk_recurrence");
    }
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
            .sequence_parallel_axis = sequence_parallel_axis,
            .num_heads = num_heads,
            .num_chunks = num_chunks,
            .key_dim = key_dim,
            .value_dim = value_dim,
            .output_bf16_mask = output_bf16_mask,
            .gate_scale = gate_scale,
            .beta_logits_column_offset = beta_logits_column_offset,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config},
        PrepareChunkRecurrenceInputs{
            .q = q, .k = k, .v = v, .g = g, .beta = beta, .actual_start = actual_start, .actual_end = actual_end});
}

}  // namespace ttnn::experimental::prim
