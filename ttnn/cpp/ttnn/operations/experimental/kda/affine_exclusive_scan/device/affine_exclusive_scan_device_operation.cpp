// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "affine_exclusive_scan_device_operation.hpp"

#include <array>
#include <vector>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"
#include "ttnn/operations/experimental/kda/kda_performance_model.hpp"

namespace ttnn::experimental::prim {

AffineExclusiveScanOperation::program_factory_t AffineExclusiveScanOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return AffineExclusiveScanProgramFactory{};
}

void AffineExclusiveScanOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    constexpr std::string_view operation_name = "affine_exclusive_scan";
    TT_FATAL(attrs.groups_per_head > 0, "affine_exclusive_scan: groups_per_head must be positive");
    TT_FATAL(
        attrs.local_rows > 0 && attrs.local_rows % tt::constants::TILE_HEIGHT == 0 &&
            (attrs.local_rows / tt::constants::TILE_HEIGHT) % attrs.groups_per_head == 0,
        "{}: local_rows must contain a positive whole number of 32-token chunks per group",
        operation_name);
    kda_factory_detail::check_actual_start(in.summaries, in.actual_start, operation_name);
    if (in.actual_end) {
        kda_factory_detail::check_actual_start(in.actual_start, *in.actual_end, "affine_exclusive_scan");
    }
    const auto summary = kda_factory_detail::check_packed_summaries(in.summaries, operation_name);
    TT_FATAL(
        in.summaries.device()->arch() == tt::ARCH::BLACKHOLE,
        "{} is only supported on Blackhole architecture, got {}",
        operation_name,
        in.summaries.device()->arch());
    kda_factory_detail::check_allocated_device_tensor(in.initial_state, operation_name, "initial_state");
    kda_factory_detail::check_layout(in.initial_state, tt::tt_metal::Layout::TILE, operation_name, "initial_state");
    kda_factory_detail::check_dtype(in.initial_state, tt::tt_metal::DataType::FLOAT32, operation_name, "initial_state");
    kda_factory_detail::check_same_device(in.summaries, in.initial_state, operation_name, "initial_state");
    kda_factory_detail::check_output_interleaved(attrs.output_mem_config, operation_name);
    kda_factory_detail::check_compute_config(attrs.compute_kernel_config, operation_name);

    TT_FATAL(
        summary.transforms == attrs.batch_heads * attrs.groups_per_head && summary.key_dim == attrs.key_dim &&
            summary.value_dim == attrs.value_dim,
        "affine_exclusive_scan: input shapes must match operation attributes");
    const auto& state_shape = in.initial_state.logical_shape();
    TT_FATAL(
        state_shape.rank() == 3 && state_shape[0] == attrs.batch_heads && state_shape[1] == attrs.key_dim &&
            state_shape[2] == attrs.value_dim,
        "affine_exclusive_scan: initial_state shape must be [batch_heads, K, V]");

    // A split can appear at runtime on any sequence-parallel mesh, so its tail transforms and seed must be present.
    const auto& mesh_shape = in.summaries.device()->shape();
    TT_FATAL(
        attrs.sequence_parallel_axis < mesh_shape.dims(),
        "{}: sequence_parallel_axis {} is outside the {}-dimensional mesh",
        operation_name,
        attrs.sequence_parallel_axis,
        mesh_shape.dims());
    TT_FATAL(
        summary.slots == 2 || mesh_shape[attrs.sequence_parallel_axis] == 1,
        "{}: summaries need a tail slot when the sequence-parallel axis has more than one device",
        operation_name);
    TT_FATAL(
        in.tail_entry_states.has_value() == (summary.slots == 2),
        "{}: tail_entry_states is required exactly when the summaries hold a tail slot",
        operation_name);
    if (in.tail_entry_states) {
        const auto& tail_entry_states = *in.tail_entry_states;
        kda_factory_detail::check_allocated_device_tensor(tail_entry_states, operation_name, "tail_entry_states");
        kda_factory_detail::check_layout(
            tail_entry_states, tt::tt_metal::Layout::TILE, operation_name, "tail_entry_states");
        kda_factory_detail::check_dtype(
            tail_entry_states, tt::tt_metal::DataType::FLOAT32, operation_name, "tail_entry_states");
        kda_factory_detail::check_same_device(in.summaries, tail_entry_states, operation_name, "tail_entry_states");
        TT_FATAL(
            tail_entry_states.logical_shape() == state_shape,
            "affine_exclusive_scan: tail_entry_states shape must match initial_state");
    }

    constexpr uint32_t max_coordinate_table_workers = 128;
    const auto grid = in.summaries.device()->compute_with_storage_grid_size();
    const uint32_t worker_limit = std::min<uint32_t>(grid.x * grid.y, max_coordinate_table_workers);
    const uint32_t group_workers = attrs.batch_heads * attrs.groups_per_head;
    TT_FATAL(
        group_workers <= worker_limit,
        "affine_exclusive_scan: supports at most {} group workers on this device, got {}",
        worker_limit,
        group_workers);
}

AffineExclusiveScanOperation::spec_return_value_t AffineExclusiveScanOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t&) {
    return {tt::tt_metal::TensorSpec(
        tt::tt_metal::Shape({a.batch_heads * a.groups_per_head, a.key_dim, a.value_dim}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::FLOAT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            a.output_mem_config))};
}
AffineExclusiveScanOperation::tensor_return_value_t AffineExclusiveScanOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    return {ttnn::create_device_tensor(compute_output_specs(a, in)[0], in.summaries.device())};
}

tt::tt_metal::operation::OpPerformanceModelGeneral<AffineExclusiveScanOperation::tensor_return_value_t>
AffineExclusiveScanOperation::create_op_performance_model(
    const operation_attributes_t& attrs, const tensor_args_t& in, tensor_return_value_t& outputs) {
    using namespace kda_performance_model;

    const double key_dim = attrs.key_dim;
    const double value_dim = attrs.value_dim;
    const double reset_transitions = attrs.batch_heads;
    const double transitions =
        static_cast<double>(attrs.batch_heads) * (attrs.groups_per_head - 1.0) + reset_transitions;
    const KdaFpuWork work{
        .fpu_matrix_flops = transitions * 2.0 * key_dim * key_dim * value_dim,
        .fpu_add_ops = transitions * key_dim * value_dim,
    };
    std::vector<const Tensor*> inputs = {&in.summaries, &in.initial_state};
    if (in.tail_entry_states) {
        inputs.push_back(&*in.tail_entry_states);
    }
    return make_profiler_model(work, inputs, outputs, attrs.compute_kernel_config.math_fidelity);
}

Tensor affine_exclusive_scan(
    const Tensor& summaries,
    const Tensor& state,
    uint32_t groups,
    const std::optional<Tensor>& tail_entry_states,
    const tt::tt_metal::MemoryConfig& mem,
    const DeviceComputeKernelConfig& cfg,
    const Tensor& actual_start,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows,
    const std::optional<Tensor>& actual_end) {
    // Cache-miss validation cannot protect attribute construction on cache hits. Keep these guards here because the
    // launcher divides by groups and indexes the summaries' shape before dispatching validation.
    TT_FATAL(groups > 0, "affine_exclusive_scan: groups_per_head must be positive");
    const auto& shape = summaries.logical_shape();
    TT_FATAL(shape.rank() == 4, "affine_exclusive_scan: summaries must be rank 4 [S, N, K, K+V]");
    TT_FATAL(shape[1] > 0, "affine_exclusive_scan: summaries must hold at least one transform");
    TT_FATAL(shape[1] % groups == 0, "affine_exclusive_scan: transforms per slot must be divisible by groups_per_head");
    TT_FATAL(shape[3] > shape[2], "affine_exclusive_scan: summary rows must be [A | B]");
    auto outputs = ::ttnn::device_operation::launch<AffineExclusiveScanOperation>(
        AffineExclusiveScanParams{
            .batch_heads = static_cast<uint32_t>(shape[1]) / groups,
            .groups_per_head = groups,
            .key_dim = static_cast<uint32_t>(shape[2]),
            .value_dim = static_cast<uint32_t>(shape[3] - shape[2]),
            .sequence_parallel_axis = sequence_parallel_axis,
            .local_rows = local_rows,
            .output_mem_config = mem,
            .compute_kernel_config = cfg},
        AffineExclusiveScanInputs{
            .summaries = summaries,
            .initial_state = state,
            .tail_entry_states = tail_entry_states,
            .actual_start = actual_start,
            .actual_end = actual_end});
    return outputs[0];
}
}  // namespace ttnn::experimental::prim
