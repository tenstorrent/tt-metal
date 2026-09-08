// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "width_to_height_shard_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

using namespace tt::constants;
using namespace tt::tt_metal;

ProgramDescriptor WidthToHeightShardProgramFactory::create_descriptor(
    const WidthToHeightShardParams& operation_attributes, const WidthToHeightShardInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    auto* device = input.device();

    const auto& input_shard = input.shard_spec().value();
    const auto& output_shard = output.shard_spec().value();
    const auto input_cores =
        corerange_to_cores(input_shard.grid, std::nullopt, input_shard.orientation == ShardOrientation::ROW_MAJOR);
    const auto output_cores = corerange_to_cores(operation_attributes.output_core_range_set, std::nullopt, true);
    const CoreRange output_bbox = operation_attributes.output_core_range_set.bounding_box();
    const CoreRangeSet output_bbox_set{output_bbox};

    const CoreCoord hub_logical = output_cores.front();
    const CoreCoord hub_physical = device->worker_core_from_logical_core(hub_logical);
    const CoreCoord mcast_start_physical = device->worker_core_from_logical_core(output_bbox.start_coord);
    const CoreCoord mcast_end_physical = device->worker_core_from_logical_core(output_bbox.end_coord);

    const auto input_format = datatype_to_dataformat_converter(input.dtype());
    const auto output_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t input_tile_size = tt::tile_size(input_format);
    const uint32_t output_tile_size = tt::tile_size(output_format);
    const uint32_t shard_height = input_shard.shape[0];
    const uint32_t shard_width = input_shard.shape[1];
    const uint32_t tiles_per_block = shard_width / TILE_WIDTH;
    const uint32_t blocks_per_core = shard_height / TILE_HEIGHT;
    const uint32_t tiles_per_core = tiles_per_block * blocks_per_core;
    const uint32_t shard_width_bytes = shard_width * output.element_size();
    const uint32_t full_width_bytes = output_shard.shape[1] * output.element_size();
    const uint32_t untilized_shard_bytes = shard_height * shard_width_bytes;
    const uint32_t full_tensor_bytes = output_shard.shape[0] * full_width_bytes;
    const uint32_t bbox_num_cores = output_bbox.grid_size().x * output_bbox.grid_size().y;

    ProgramDescriptor desc;

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_untilized = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t gather_semaphore_id = 0;

    desc.cbs.push_back(CBDescriptor{
        .total_size = tiles_per_core * input_tile_size,
        .core_ranges = input_shard.grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_in),
            .data_format = input_format,
            .page_size = input_tile_size,
            .tile = input.tensor_spec().tile(),
        }}},
        .buffer = input.buffer(),
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = untilized_shard_bytes,
        .core_ranges = input_shard.grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_untilized),
            .data_format = output_format,
            .page_size = output_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = full_tensor_bytes,
        .core_ranges = output_shard.grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_out),
            .data_format = output_format,
            .page_size = full_width_bytes,
        }}},
        .buffer = output.buffer(),
    });

    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = gather_semaphore_id,
        .core_ranges = output_bbox_set,
        .initial_value = 0,
    });

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek/width_to_height_shard/device/kernels/dataflow/"
        "writer_width_to_height_shard.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = input_shard.grid;
    writer_desc.compile_time_args = {
        cb_in,
        cb_untilized,
        tiles_per_core,
        output_shard.shape[0],
        shard_width_bytes,
        full_width_bytes,
        gather_semaphore_id,
    };
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = input_shard.grid;
    compute_desc.compile_time_args = {blocks_per_core, tiles_per_block, cb_in, cb_untilized};
    if (input.dtype() == DataType::INT32 || input.dtype() == DataType::UINT32 || input.dtype() == DataType::FLOAT32) {
        compute_desc.defines = {{"DST_ACCUM_MODE", "1"}};
    }
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en =
            input.dtype() == DataType::INT32 || input.dtype() == DataType::UINT32 || input.dtype() == DataType::FLOAT32,
    };

    KernelDescriptor receiver_desc;
    receiver_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek/width_to_height_shard/device/kernels/dataflow/"
        "reader_width_to_height_shard.cpp";
    receiver_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    receiver_desc.core_ranges = output_bbox_set;
    receiver_desc.compile_time_args = {
        static_cast<uint32_t>(input_cores.size()),
        full_tensor_bytes,
        bbox_num_cores,
        static_cast<uint32_t>(mcast_start_physical.x),
        static_cast<uint32_t>(mcast_start_physical.y),
        static_cast<uint32_t>(mcast_end_physical.x),
        static_cast<uint32_t>(mcast_end_physical.y),
        gather_semaphore_id,
    };
    receiver_desc.config = ReaderConfigDescriptor{};

    for (uint32_t sender_id = 0; sender_id < input_cores.size(); ++sender_id) {
        writer_desc.emplace_runtime_args(
            input_cores[sender_id],
            {sender_id, output.buffer(), static_cast<uint32_t>(hub_physical.x), static_cast<uint32_t>(hub_physical.y)});
    }

    const auto bbox_cores = corerange_to_cores(output_bbox_set, std::nullopt, true);
    for (const auto& core : bbox_cores) {
        receiver_desc.emplace_runtime_args(core, {static_cast<uint32_t>(core == hub_logical), output.buffer()});
    }

    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    desc.kernels.push_back(std::move(receiver_desc));
    return desc;
}

}  // namespace ttnn::prim
