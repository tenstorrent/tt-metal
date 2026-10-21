// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "entry_correction_device_operation.hpp"

#include <array>
#include <set>
#include <utility>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"
#include "ttnn/operations/experimental/kda/kda_performance_model.hpp"

namespace ttnn::experimental::prim {

EntryCorrectionOperation::program_factory_t EntryCorrectionOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return EntryCorrectionProgramFactory{};
}

void EntryCorrectionOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    constexpr std::string_view operation_name = "entry_correction";
    TT_FATAL(
        attrs.local_rows > 0 && attrs.local_rows % tt::constants::TILE_HEIGHT == 0,
        "{}: local_rows must be positive and 32-aligned",
        operation_name);
    kda_factory_detail::check_actual_start(in.final_b, in.actual_start, operation_name);
    if (in.actual_end) {
        kda_factory_detail::check_actual_start(in.actual_start, *in.actual_end, operation_name);
    }
    for (const auto& [tensor, name] :
         {std::pair{&in.final_a, "final_a"},
          std::pair{&in.final_b, "final_b"},
          std::pair{&in.entry, "entry"},
          std::pair{&in.tail_entry, "tail_entry"}}) {
        kda_factory_detail::check_allocated_device_tensor(*tensor, operation_name, name);
        kda_factory_detail::check_layout(*tensor, tt::tt_metal::Layout::TILE, operation_name, name);
        kda_factory_detail::check_dtype(*tensor, tt::tt_metal::DataType::FLOAT32, operation_name, name);
        kda_factory_detail::check_interleaved(*tensor, operation_name, name);
        kda_factory_detail::check_same_device(in.final_b, *tensor, operation_name, name);
        TT_FATAL(
            tensor->logical_shape() == in.final_b.logical_shape(),
            "{}: {} must have final_b's shape",
            operation_name,
            name);
    }
    const auto& state = in.final_b.logical_shape();
    // The kernels read every state and chunk row as K tiles: V must equal K.
    TT_FATAL(state.rank() == 3 && state[1] == state[2], "{}: the states must be [B*H, K, K]", operation_name);
    TT_FATAL(
        state[0] > 0 && state[1] > 0 && state[1] % tt::constants::TILE_WIDTH == 0,
        "{}: B*H must be positive and K positive and tile aligned",
        operation_name);
    const uint32_t heads = state[0];
    const uint32_t key_dim = state[1];
    const uint32_t chunks = attrs.local_rows / tt::constants::TILE_HEIGHT;
    TT_FATAL(
        !attrs.cores.empty() && attrs.cores.size() % heads == 0 && chunks % (attrs.cores.size() / heads) == 0,
        "{}: the cores must split evenly over the {} heads, and a head's cores over its {} chunks",
        operation_name,
        heads,
        chunks);
    const uint32_t cores_per_head = attrs.cores.size() / heads;
    const auto* mesh = in.final_b.device();
    const auto grid = mesh->compute_with_storage_grid_size();
    TT_FATAL(
        std::set<tt::tt_metal::CoreCoord>(attrs.cores.begin(), attrs.cores.end()).size() == attrs.cores.size(),
        "{}: the cores must be distinct",
        operation_name);
    for (const auto& core : attrs.cores) {
        TT_FATAL(
            core.x < grid.x && core.y < grid.y,
            "{}: core {} is outside the {}x{} compute grid",
            operation_name,
            core,
            grid.x,
            grid.y);
    }
    for (const auto& [terms, name] :
         {std::pair{&in.zero_output, "zero_output"}, std::pair{&in.identity_output, "identity_output"}}) {
        kda_factory_detail::check_allocated_device_tensor(*terms, operation_name, name);
        kda_factory_detail::check_layout(*terms, tt::tt_metal::Layout::TILE, operation_name, name);
        kda_factory_detail::check_dtype(*terms, tt::tt_metal::DataType::FLOAT32, operation_name, name);
        kda_factory_detail::check_same_device(in.final_b, *terms, operation_name, name);
        TT_FATAL(
            terms->logical_shape() == Shape({heads, chunks, tt::constants::TILE_HEIGHT, key_dim}),
            "{}: {} must be [B*H, {}, 32, K]",
            operation_name,
            name,
            chunks);
        const auto& memory = terms->memory_config();
        TT_FATAL(
            memory.buffer_type() == tt::tt_metal::BufferType::L1 &&
                memory.memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            "{}: {} must be height-sharded in L1",
            operation_name,
            name);
        const auto& shard = *memory.shard_spec();
        TT_FATAL(
            shard.shape[0] == chunks / cores_per_head * tt::constants::TILE_HEIGHT && shard.shape[1] == key_dim,
            "{}: each {} shard must be one core's {} chunk rows",
            operation_name,
            name,
            chunks / cores_per_head);
        // Core i reads shard i in place.
        TT_FATAL(
            kda_factory_detail::shard_cores(*terms) == attrs.cores,
            "{}: {} must be sharded over the cores in their order",
            operation_name,
            name);
    }
    kda_factory_detail::check_output_interleaved(attrs.output_mem_config, operation_name);
    kda_factory_detail::check_compute_config(attrs.compute_kernel_config, operation_name);
    // Compute packs a K-tile row of each chunk output and of the final state out of DST.
    const uint32_t key_tiles = key_dim / tt::constants::TILE_WIDTH;
    const uint32_t dst_tiles = ttnn::get_dest_reg_count(attrs.compute_kernel_config);
    TT_FATAL(
        key_tiles <= dst_tiles,
        "{}: K = {} tiles exceeds the {} DST tiles a row packs from",
        operation_name,
        key_tiles,
        dst_tiles);
    TT_FATAL(attrs.sequence_parallel_axis < mesh->shape().dims(), "{}: invalid sequence_parallel_axis", operation_name);
}

EntryCorrectionOperation::spec_return_value_t EntryCorrectionOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const auto spec =
        [](const ttnn::Shape& shape, tt::tt_metal::DataType dtype, const tt::tt_metal::MemoryConfig& memory) {
            return tt::tt_metal::TensorSpec(
                shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), memory));
        };
    return {
        spec(in.zero_output.logical_shape(), tt::tt_metal::DataType::BFLOAT16, attrs.output_mem_config),
        spec(in.final_b.logical_shape(), tt::tt_metal::DataType::FLOAT32, ttnn::DRAM_MEMORY_CONFIG)};
}

EntryCorrectionOperation::tensor_return_value_t EntryCorrectionOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(attrs, in)) {
        outputs.push_back(create_device_tensor(spec, in.final_b.device()));
    }
    return outputs;
}

tt::tt_metal::operation::OpPerformanceModelGeneral<EntryCorrectionOperation::tensor_return_value_t>
EntryCorrectionOperation::create_op_performance_model(
    const operation_attributes_t& attrs, const tensor_args_t& in, tensor_return_value_t& outputs) {
    using namespace kda_performance_model;
    const auto& state = in.final_b.logical_shape();
    const double heads = state[0];
    const double key_dim = state[1];
    const double rows = attrs.local_rows;
    // Per head: the chunk rows and the final state each multiply a [rows, K] or [K, K] block by the K x K seed, after
    // the chunk rows' differences FI - F0.
    const KdaFpuWork work{
        .fpu_matrix_flops = heads * 2.0 * (rows + key_dim) * key_dim * key_dim,
        .fpu_add_ops = heads * rows * key_dim,
    };
    const std::array<const Tensor*, 6> inputs = {
        &in.zero_output, &in.identity_output, &in.final_a, &in.final_b, &in.entry, &in.tail_entry};
    return make_profiler_model(work, inputs, outputs, attrs.compute_kernel_config.math_fidelity);
}

std::tuple<Tensor, Tensor> entry_correction(
    const Tensor& zero_output,
    const Tensor& identity_output,
    const Tensor& final_a,
    const Tensor& final_b,
    const Tensor& entry,
    const Tensor& tail_entry,
    const tt::tt_metal::MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const Tensor& actual_start,
    const std::optional<Tensor>& actual_end,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows,
    const std::vector<tt::tt_metal::CoreCoord>& cores) {
    auto outputs = ttnn::device_operation::launch<EntryCorrectionOperation>(
        EntryCorrectionParams{
            .sequence_parallel_axis = sequence_parallel_axis,
            .local_rows = local_rows,
            .cores = cores,
            .output_mem_config = memory_config,
            .compute_kernel_config = compute_kernel_config},
        EntryCorrectionInputs{
            .zero_output = zero_output,
            .identity_output = identity_output,
            .final_a = final_a,
            .final_b = final_b,
            .entry = entry,
            .tail_entry = tail_entry,
            .actual_start = actual_start,
            .actual_end = actual_end});
    return {outputs[0], outputs[1]};
}

}  // namespace ttnn::experimental::prim
