// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "relay_affine_prefix_device_operation.hpp"

#include <array>
#include <set>
#include <span>
#include <utility>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_projection.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"
#include "ttnn/operations/experimental/kda/kda_performance_model.hpp"

namespace ttnn::experimental::prim {

RelayAffinePrefixOperation::program_factory_t RelayAffinePrefixOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return RelayAffinePrefixProgramFactory{};
}

void RelayAffinePrefixOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    constexpr std::string_view operation_name = "relay_affine_prefix";
    using tt::tt_metal::CoreCoord;
    TT_FATAL(
        attrs.local_rows > 0 && attrs.local_rows % tt::constants::TILE_HEIGHT == 0,
        "{}: local_rows must be positive and 32-aligned",
        operation_name);
    kda_factory_detail::check_actual_start(in.packed, in.actual_start, operation_name);
    kda_factory_detail::check_allocated_device_tensor(in.packed, operation_name, "packed");
    kda_factory_detail::check_layout(in.packed, tt::tt_metal::Layout::TILE, operation_name, "packed");
    kda_factory_detail::check_dtype(in.packed, tt::tt_metal::DataType::BFLOAT16, operation_name, "packed");
    kda_factory_detail::check_interleaved(in.packed, operation_name, "packed");
    kda_factory_detail::check_allocated_device_tensor(in.initial_state, operation_name, "initial_state");
    kda_factory_detail::check_layout(in.initial_state, tt::tt_metal::Layout::TILE, operation_name, "initial_state");
    kda_factory_detail::check_dtype(in.initial_state, tt::tt_metal::DataType::FLOAT32, operation_name, "initial_state");
    kda_factory_detail::check_interleaved(in.initial_state, operation_name, "initial_state");
    kda_factory_detail::check_same_device(in.packed, in.initial_state, operation_name, "initial_state");
    const auto& state = in.initial_state.logical_shape();
    TT_FATAL(state.rank() == 3 && state[1] == state[2], "{}: initial_state must be [B*H, K, K]", operation_name);
    TT_FATAL(
        state[0] > 0 && state[1] > 0 && state[1] % tt::constants::TILE_WIDTH == 0,
        "{}: B*H must be positive and K positive and tile aligned",
        operation_name);
    const uint32_t heads = state[0];
    const uint32_t key_dim = state[1];
    const uint32_t key_tiles = key_dim / tt::constants::TILE_WIDTH;
    // The reader indexes packed as B*H x K tile rows of [A | B].
    TT_FATAL(
        in.packed.padded_shape()[-1] == 2 * key_dim && in.packed.physical_volume() == heads * key_dim * 2 * key_dim,
        "{}: packed must be [B*H, K, 2K] rows of [A | B]",
        operation_name);

    const auto* mesh = in.packed.device();
    TT_FATAL(attrs.sequence_parallel_axis < mesh->shape().dims(), "{}: invalid sequence_parallel_axis", operation_name);
    TT_FATAL(
        mesh->shape()[attrs.sequence_parallel_axis] > 2, "{}: needs a ring of more than two SP ranks", operation_name);
    TT_FATAL(
        attrs.num_links > 0 && attrs.cores.size() >= heads + attrs.num_links,
        "{}: needs a core per head, then a sender per link",
        operation_name);
    const std::span<const CoreCoord> used_cores(attrs.cores.data(), heads + attrs.num_links);
    TT_FATAL(
        std::set<CoreCoord>(used_cores.begin(), used_cores.end()).size() == used_cores.size(),
        "{}: the head and sender cores must be distinct",
        operation_name);
    const auto grid = mesh->compute_with_storage_grid_size();
    for (const auto& core : used_cores) {
        TT_FATAL(
            core.x < grid.x && core.y < grid.y,
            "{}: core {} is outside the {}x{} compute grid",
            operation_name,
            core,
            grid.x,
            grid.y);
    }
    TT_FATAL(
        attrs.semaphore_addresses[0] != 0 && attrs.semaphore_addresses[1] != 0,
        "{}: needs two global semaphores",
        operation_name);
    TT_FATAL(
        tt::tt_fabric::get_tt_fabric_max_payload_size_bytes() >= tt::constants::TILE_HW * sizeof(float),
        "{}: the relay sends whole FP32 tiles",
        operation_name);
    TT_FATAL(
        relay_affine_prefix_sender_slots(key_tiles, mesh->arch()) > 0,
        "{}: a tile column of K = {} tiles needs {} packet headers, the pool holds {} per RISC",
        operation_name,
        key_tiles,
        relay_affine_prefix_column_packets(key_tiles),
        fabric_packet_headers_per_risc(mesh->arch()));

    const std::span<const CoreCoord> head_cores(attrs.cores.data(), heads);
    for (const auto& [landing, name] :
         {std::pair{&in.entry_landing, "entry_landing"},
          std::pair{&in.state_staging, "state_staging"},
          std::pair{&in.final_landing, "final_landing"}}) {
        kda_factory_detail::check_allocated_device_tensor(*landing, operation_name, name);
        kda_factory_detail::check_layout(*landing, tt::tt_metal::Layout::TILE, operation_name, name);
        kda_factory_detail::check_dtype(*landing, tt::tt_metal::DataType::FLOAT32, operation_name, name);
        kda_factory_detail::check_same_device(in.packed, *landing, operation_name, name);
        const auto& memory = landing->memory_config();
        TT_FATAL(
            memory.buffer_type() == tt::tt_metal::BufferType::L1 &&
                memory.memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            "{}: {} must be height-sharded in L1",
            operation_name,
            name);
        const auto& shard = *memory.shard_spec();
        TT_FATAL(
            landing->physical_volume() == heads * key_dim * key_dim && shard.shape[0] == key_dim &&
                shard.shape[1] == key_dim && kda_factory_detail::same_cores(shard.grid, head_cores),
            "{}: {} must hold one {} x {} state per head core",
            operation_name,
            name,
            key_dim,
            key_dim);
    }
    kda_factory_detail::check_output_interleaved(attrs.entry_mem_config, operation_name);
    kda_factory_detail::check_compute_config(attrs.compute_kernel_config, operation_name);
    // Compute packs a K-tile column of the next state out of DST.
    const uint32_t dst_tiles = ttnn::get_dest_reg_count(attrs.compute_kernel_config);
    TT_FATAL(
        key_tiles <= dst_tiles,
        "{}: K = {} tiles exceeds the {} DST tiles a column packs from",
        operation_name,
        key_tiles,
        dst_tiles);
    if (attrs.output_gate) {
        const auto& gate = *attrs.output_gate;
        TT_FATAL(in.gate_input && in.gate_weight, "{}: an output gate needs its input and weight", operation_name);
        kda_factory_detail::check_projection(*in.gate_input, *in.gate_weight, gate.program_config, operation_name);
        TT_FATAL(
            gate.output_column == 0 || (in.gate_copy_source && !gate.copy_cores.empty()),
            "{}: the output gate's copied columns need a copy source and copy cores",
            operation_name);
    }
}

RelayAffinePrefixOperation::spec_return_value_t RelayAffinePrefixOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const auto spec = [&](const tt::tt_metal::MemoryConfig& memory) {
        return tt::tt_metal::TensorSpec(
            in.initial_state.logical_shape(),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::FLOAT32, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), memory));
    };
    spec_return_value_t specs = {spec(attrs.entry_mem_config), spec(ttnn::DRAM_MEMORY_CONFIG)};
    if (attrs.output_gate) {
        // The copied tile columns, then the projection's.
        auto shape = in.gate_input->logical_shape();
        shape[-1] = attrs.output_gate->output_column * tt::constants::TILE_WIDTH + in.gate_weight->logical_shape()[-1];
        specs.emplace_back(
            shape,
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                attrs.output_gate->output_mem_config));
    }
    return specs;
}

RelayAffinePrefixOperation::tensor_return_value_t RelayAffinePrefixOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(attrs, in)) {
        outputs.push_back(create_device_tensor(spec, in.packed.device()));
    }
    return outputs;
}

tt::tt_metal::operation::OpPerformanceModelGeneral<RelayAffinePrefixOperation::tensor_return_value_t>
RelayAffinePrefixOperation::create_op_performance_model(
    const operation_attributes_t& attrs, const tensor_args_t& in, tensor_return_value_t& outputs) {
    using namespace kda_performance_model;
    const auto& state = in.initial_state.logical_shape();
    const double heads = state[0];
    const double key_dim = state[1];
    // One hop per rank: E_out = A E_in + B per head.
    const KdaFpuWork work{
        .fpu_matrix_flops = heads * 2.0 * key_dim * key_dim * key_dim,
        .fpu_add_ops = heads * key_dim * key_dim,
    };
    const std::array<const Tensor*, 2> inputs = {&in.packed, &in.initial_state};
    return make_profiler_model(work, inputs, outputs, attrs.compute_kernel_config.math_fidelity);
}

std::tuple<Tensor, Tensor, std::optional<Tensor>> relay_affine_prefix(
    const Tensor& packed,
    const Tensor& initial_state,
    const Tensor& actual_start,
    const Tensor& entry_landing,
    const Tensor& state_staging,
    const Tensor& final_landing,
    const tt::tt_metal::MemoryConfig& entry_memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const std::array<uint32_t, 2>& semaphore_addresses,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows,
    uint32_t num_links,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    const std::optional<AffineRelayOutputGateParams>& output_gate,
    const std::optional<Tensor>& gate_input,
    const std::optional<Tensor>& gate_weight,
    const std::optional<Tensor>& gate_copy_source) {
    auto outputs = ttnn::device_operation::launch<RelayAffinePrefixOperation>(
        RelayAffinePrefixParams{
            .sequence_parallel_axis = sequence_parallel_axis,
            .local_rows = local_rows,
            .num_links = num_links,
            .cores = cores,
            .entry_mem_config = entry_memory_config,
            .compute_kernel_config = compute_kernel_config,
            .semaphore_addresses = semaphore_addresses,
            .output_gate = output_gate},
        RelayAffinePrefixInputs{
            .packed = packed,
            .initial_state = initial_state,
            .actual_start = actual_start,
            .entry_landing = entry_landing,
            .state_staging = state_staging,
            .final_landing = final_landing,
            .gate_input = gate_input,
            .gate_weight = gate_weight,
            .gate_copy_source = gate_copy_source});
    return {outputs[0], outputs[1], outputs.size() > 2 ? std::optional<Tensor>(outputs[2]) : std::nullopt};
}

}  // namespace ttnn::experimental::prim
