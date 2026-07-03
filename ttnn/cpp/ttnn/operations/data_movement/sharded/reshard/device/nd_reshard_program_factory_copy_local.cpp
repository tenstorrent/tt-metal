// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/reshard/device/nd_reshard_program_factory_copy_local.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "tt-metalium/host_api.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

template <bool local_is_input>
ProgramDescriptor NdReshardCopyLocalShardFactory<local_is_input>::create_descriptor(
    const ReshardParams& /*operation_attributes*/, const ReshardInputs& tensor_args, Tensor& output_tensor) {
    const auto& input = tensor_args.input;
    auto& output = output_tensor;

    auto* input_buffer = input.buffer();
    auto* output_buffer = output.buffer();

    const auto input_accessor_args = TensorAccessorArgs(*input_buffer);
    const auto output_accessor_args = TensorAccessorArgs(*output_buffer);

    // Choose buffer and aligned page size based on local_is_input flag
    auto* local_buffer = local_is_input ? input_buffer : output_buffer;
    auto aligned_page_size = local_buffer->aligned_page_size();
    auto other_aligned_page_size =
        local_is_input ? output_buffer->aligned_page_size() : input_buffer->aligned_page_size();

    // This implementation assumes that input and output grids are the same.
    auto cores_vec = local_buffer->buffer_distribution_spec()->cores_with_data();
    auto grid = CoreRangeSet(cores_vec);

    uint32_t num_shards = static_cast<uint32_t>(local_buffer->buffer_distribution_spec()->num_shards());

    // num cores with data * 2 because we have two kernels
    uint32_t shard_id_stride =
        static_cast<uint32_t>(local_buffer->buffer_distribution_spec()->num_cores_with_data()) * 2u;

    // Prepare compile time arguments
    auto logical_size = input.logical_shape();
    uint32_t logical_width = static_cast<uint32_t>(logical_size[-1] * input.element_size());
    uint32_t source_width = logical_width;
    uint32_t destination_width = logical_width;
    uint32_t base_page_size = aligned_page_size;

    if (input.memory_config().shard_spec().has_value() && output.memory_config().shard_spec().has_value()) {
        auto input_buffer_type = input.memory_config().memory_layout();
        auto output_buffer_type = output.memory_config().memory_layout();

        // for block sharded
        CoreCoord input_shard_grid = input_buffer->shard_spec().grid().ranges()[0].grid_size();
        uint32_t input_num_shard_cores = input_shard_grid.x;
        if (input_buffer->shard_spec().orientation() == ShardOrientation::COL_MAJOR) {
            input_num_shard_cores = input_shard_grid.y;
        }

        CoreCoord output_shard_grid = output_buffer->shard_spec().grid().ranges()[0].grid_size();
        uint32_t output_num_shard_cores = output_shard_grid.x;
        if (output_buffer->shard_spec().orientation() == ShardOrientation::COL_MAJOR) {
            output_num_shard_cores = output_shard_grid.y;
        }
        // for width sharded
        if (input_buffer_type == TensorMemoryLayout::WIDTH_SHARDED &&
            output_buffer_type == TensorMemoryLayout::WIDTH_SHARDED) {
            input_num_shard_cores = input_shard_grid.x == 1 ? input_shard_grid.y : input_shard_grid.x;
            output_num_shard_cores = output_shard_grid.x == 1 ? output_shard_grid.y : output_shard_grid.x;
        }

        source_width =
            static_cast<uint32_t>(input_buffer->shard_spec().shape()[1] * input.element_size() * input_num_shard_cores);
        destination_width = static_cast<uint32_t>(
            output_buffer->shard_spec().shape()[1] * output.element_size() * output_num_shard_cores);
        uint32_t input_page_size = input_buffer->page_size();
        uint32_t output_page_size = output_buffer->page_size();
        base_page_size = std::gcd(input_page_size, output_page_size);
    }
    auto compile_time_args = input_accessor_args.get_compile_time_args();
    output_accessor_args.append_to(compile_time_args);
    compile_time_args.push_back(aligned_page_size);
    compile_time_args.push_back(other_aligned_page_size);
    compile_time_args.push_back(static_cast<uint32_t>(local_is_input));
    compile_time_args.push_back(logical_width);
    compile_time_args.push_back(source_width);
    compile_time_args.push_back(destination_width);
    compile_time_args.push_back(base_page_size);

    ProgramDescriptor desc;

    KernelDescriptor brisc_desc;
    brisc_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    brisc_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/nd_reshard_copy_local_shards.cpp";
    brisc_desc.core_ranges = grid;
    brisc_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
    };
    brisc_desc.compile_time_args = compile_time_args;

    KernelDescriptor ncrisc_desc;
    ncrisc_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    ncrisc_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/nd_reshard_copy_local_shards.cpp";
    ncrisc_desc.core_ranges = grid;
    ncrisc_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
    };
    ncrisc_desc.compile_time_args = std::move(compile_time_args);

    // Common runtime args: [input_addr, output_addr, num_shards, shard_id_stride]
    // arg 0 / arg 1 are the buffer base addresses (binding via Buffer*).
    brisc_desc.emplace_common_runtime_args({input_buffer, output_buffer, num_shards, shard_id_stride});
    ncrisc_desc.emplace_common_runtime_args({input_buffer, output_buffer, num_shards, shard_id_stride});

    // Per-core unique runtime args: [start_shard_id]
    // brisc copies shards [0, num_data_cores*2, num_data_cores*4, num_data_cores*6, ...]
    // ncrisc copies shards [num_data_cores, num_data_cores*3, num_data_cores*5, num_data_cores*7, ...]
    uint32_t start_shard_id = 0;
    for (const auto& core : cores_vec) {
        brisc_desc.emplace_runtime_args(core, {start_shard_id});
        ncrisc_desc.emplace_runtime_args(core, {start_shard_id + shard_id_stride / 2});
        ++start_shard_id;
    }

    desc.kernels.push_back(std::move(brisc_desc));
    desc.kernels.push_back(std::move(ncrisc_desc));

    return desc;
}

// Explicit template instantiations
template struct NdReshardCopyLocalShardFactory<true>;
template struct NdReshardCopyLocalShardFactory<false>;

}  // namespace ttnn::prim
