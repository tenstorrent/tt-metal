// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_for_matmul_program_factory.hpp"
#include "all_gather_for_matmul_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

ProgramDescriptor create_gather_descriptor(
    const AllGatherForMatmulParams& operation_attributes, const AllGatherForMatmulInputs& tensor_args, Tensor& output) {
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
    uint32_t full_width_bytes = output_shard.shape[1] * output.element_size();
    uint32_t untilized_shard_bytes = shard_height * shard_width * output.element_size();
    uint32_t full_tensor_bytes = output_shard.shape[0] * full_width_bytes;
    const uint32_t bbox_num_cores = output_bbox.grid_size().x * output_bbox.grid_size().y;
    // Row tiles (h x 32, one tile row): a core's width shard is a contiguous run of the replica's
    // tiles, so it is appended as is (no untilize, one write per sender).
    const bool row_tiles = all_gather_for_matmul_row_tiles(input);
    const bool skip_untilize = input.layout() == Layout::ROW_MAJOR || row_tiles;
    // Height shards are consecutive full-width row blocks, so each one lands in the replica as a
    // single contiguous run at sender_id * shard bytes.
    const bool stacked_rows = input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED || row_tiles;
    const uint32_t input_row_tile_size = input.tensor_spec().tile().get_tile_size(input_format);
    if (row_tiles) {
        untilized_shard_bytes =
            (shard_height / input.tensor_spec().tile().get_tile_shape()[0]) * tiles_per_block * input_row_tile_size;
        full_tensor_bytes = untilized_shard_bytes * input_cores.size();
        full_width_bytes = full_tensor_bytes;
    }
    const uint32_t shard_width_bytes = stacked_rows ? untilized_shard_bytes : shard_width * output.element_size();
    const uint32_t rows_per_slice = stacked_rows ? 1 : input.logical_shape()[-2];
    const uint32_t padded_rows_per_slice = stacked_rows ? 1 : input.padded_shape()[-2];
    const uint32_t num_slices = stacked_rows ? 1 : output_shard.shape[0] / rows_per_slice;

    ProgramDescriptor desc;

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_untilized = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t gather_semaphore_id = 0;

    if (skip_untilize) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = untilized_shard_bytes,
            .core_ranges = input_shard.grid,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_in),
                .data_format = output_format,
                .page_size = shard_width_bytes,
            }}},
            .buffer = input.buffer(),
        });
    } else {
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
    }

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
        "ttnn/cpp/ttnn/operations/experimental/deepseek/all_gather_for_matmul/device/kernels/dataflow/"
        "writer_all_gather_for_matmul.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = input_shard.grid;
    writer_desc.compile_time_args = {
        cb_in,
        cb_untilized,
        tiles_per_core,
        static_cast<uint32_t>(skip_untilize),
        num_slices,
        rows_per_slice,
        padded_rows_per_slice,
        shard_width_bytes,
        full_width_bytes,
        gather_semaphore_id,
    };
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor receiver_desc;
    receiver_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek/all_gather_for_matmul/device/kernels/dataflow/"
        "reader_all_gather_for_matmul.cpp";
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
    if (!skip_untilize) {
        KernelDescriptor compute_desc;
        compute_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp";
        compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc.core_ranges = input_shard.grid;
        compute_desc.compile_time_args = {blocks_per_core, tiles_per_block, cb_in, cb_untilized};
        if (input.dtype() == DataType::INT32 || input.dtype() == DataType::UINT32 ||
            input.dtype() == DataType::FLOAT32) {
            compute_desc.defines = {{"DST_ACCUM_MODE", "1"}};
        }
        compute_desc.config = ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = input.dtype() == DataType::INT32 || input.dtype() == DataType::UINT32 ||
                                input.dtype() == DataType::FLOAT32,
        };
        desc.kernels.push_back(std::move(compute_desc));
    }
    desc.kernels.push_back(std::move(receiver_desc));
    return desc;
}

ProgramDescriptor create_mcast_descriptor(
    const AllGatherForMatmulParams& operation_attributes, const AllGatherForMatmulInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    auto* device = input.device();

    const auto& input_shard = input.shard_spec().value();
    const auto& output_shard = output.shard_spec().value();
    const auto input_cores =
        corerange_to_cores(input_shard.grid, std::nullopt, input_shard.orientation == ShardOrientation::ROW_MAJOR);
    const auto output_cores = corerange_to_cores(operation_attributes.output_core_range_set, std::nullopt, true);
    const CoreRange output_bbox = operation_attributes.output_core_range_set.bounding_box();
    const CoreRangeSet output_bbox_set{output_bbox};
    const CoreCoord source_logical = input_cores.front();
    const CoreCoord mcast_start_physical = device->worker_core_from_logical_core(output_bbox.start_coord);
    const CoreCoord mcast_end_physical = device->worker_core_from_logical_core(output_bbox.end_coord);

    // The multicast must be issued from inside the destination rectangle. When the source core
    // already holds an output shard it is its own hub; otherwise it hands the replica to a core
    // in the output set and that core multicasts, matching the gather path.
    const bool source_is_hub = operation_attributes.output_core_range_set.contains(source_logical);
    const CoreCoord hub_logical = source_is_hub ? source_logical : output_cores.front();
    const CoreCoord hub_physical = device->worker_core_from_logical_core(hub_logical);

    const auto input_format = datatype_to_dataformat_converter(input.dtype());
    const auto output_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t input_tile_size = tt::tile_size(input_format);
    const uint32_t output_tile_size = tt::tile_size(output_format);
    const uint32_t shard_height = input_shard.shape[0];
    const uint32_t shard_width = input_shard.shape[1];
    const uint32_t tiles_per_block = shard_width / TILE_WIDTH;
    const uint32_t blocks_per_core = shard_height / TILE_HEIGHT;
    const uint32_t tiles_per_core = tiles_per_block * blocks_per_core;
    uint32_t full_width_bytes = output_shard.shape[1] * output.element_size();
    uint32_t untilized_shard_bytes = shard_height * shard_width * output.element_size();
    uint32_t full_tensor_bytes = output_shard.shape[0] * full_width_bytes;
    const uint32_t bbox_num_cores = output_bbox.grid_size().x * output_bbox.grid_size().y;
    // Row tiles (h x 32): the shard already is the replica's bytes; multicast it as one page.
    const bool row_tiles = all_gather_for_matmul_row_tiles(input);
    const bool skip_untilize = input.layout() == Layout::ROW_MAJOR || row_tiles;
    if (row_tiles) {
        full_tensor_bytes = (shard_height / input.tensor_spec().tile().get_tile_shape()[0]) * tiles_per_block *
                            input.tensor_spec().tile().get_tile_size(input_format);
        full_width_bytes = full_tensor_bytes;
    }
    const uint32_t rows_per_slice = row_tiles ? 1 : input.logical_shape()[-2];
    const uint32_t padded_rows_per_slice = row_tiles ? 1 : input.padded_shape()[-2];
    const uint32_t num_slices = row_tiles ? 1 : output_shard.shape[0] / rows_per_slice;

    ProgramDescriptor desc;

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_untilized = tt::CBIndex::c_1;
    constexpr uint32_t gather_semaphore_id = 0;

    if (skip_untilize) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = row_tiles ? full_tensor_bytes : shard_height * full_width_bytes,
            .core_ranges = input_shard.grid,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_in),
                .data_format = output_format,
                .page_size = full_width_bytes,
            }}},
            .buffer = input.buffer(),
        });
    } else {
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
    }

    desc.cbs.push_back(CBDescriptor{
        .total_size = full_tensor_bytes,
        .core_ranges = output_shard.grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_16),
            .data_format = output_format,
            .page_size = full_width_bytes,
        }}},
        .buffer = output.buffer(),
    });

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek/all_gather_for_matmul/device/kernels/dataflow/"
        "writer_mcast_for_matmul.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = input_shard.grid;
    writer_desc.compile_time_args = {
        cb_in,
        cb_untilized,
        tiles_per_core,
        static_cast<uint32_t>(skip_untilize),
        num_slices,
        rows_per_slice,
        padded_rows_per_slice,
        full_width_bytes,
        bbox_num_cores,
        static_cast<uint32_t>(mcast_start_physical.x),
        static_cast<uint32_t>(mcast_start_physical.y),
        static_cast<uint32_t>(mcast_end_physical.x),
        static_cast<uint32_t>(mcast_end_physical.y),
        static_cast<uint32_t>(source_is_hub),
        gather_semaphore_id,
    };
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.emplace_runtime_args(
        source_logical,
        {output.buffer(), static_cast<uint32_t>(hub_physical.x), static_cast<uint32_t>(hub_physical.y)});

    desc.kernels.push_back(std::move(writer_desc));

    if (!skip_untilize) {
        KernelDescriptor compute_desc;
        compute_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp";
        compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc.core_ranges = input_shard.grid;
        compute_desc.compile_time_args = {blocks_per_core, tiles_per_block, cb_in, cb_untilized};
        if (input.dtype() == DataType::INT32 || input.dtype() == DataType::UINT32 ||
            input.dtype() == DataType::FLOAT32) {
            compute_desc.defines = {{"DST_ACCUM_MODE", "1"}};
        }
        compute_desc.config = ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = input.dtype() == DataType::INT32 || input.dtype() == DataType::UINT32 ||
                                input.dtype() == DataType::FLOAT32,
        };
        desc.kernels.push_back(std::move(compute_desc));
    }

    if (!source_is_hub) {
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = gather_semaphore_id,
            .core_ranges = output_bbox_set,
            .initial_value = 0,
        });

        KernelDescriptor receiver_desc;
        receiver_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek/all_gather_for_matmul/device/kernels/dataflow/"
            "reader_all_gather_for_matmul.cpp";
        receiver_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        receiver_desc.core_ranges = output_bbox_set;
        receiver_desc.compile_time_args = {
            1,  // a single sender: the source core owns the whole tensor
            full_tensor_bytes,
            bbox_num_cores,
            static_cast<uint32_t>(mcast_start_physical.x),
            static_cast<uint32_t>(mcast_start_physical.y),
            static_cast<uint32_t>(mcast_end_physical.x),
            static_cast<uint32_t>(mcast_end_physical.y),
            gather_semaphore_id,
        };
        receiver_desc.config = ReaderConfigDescriptor{};

        const auto bbox_cores = corerange_to_cores(output_bbox_set, std::nullopt, true);
        for (const auto& core : bbox_cores) {
            receiver_desc.emplace_runtime_args(core, {static_cast<uint32_t>(core == hub_logical), output.buffer()});
        }
        desc.kernels.push_back(std::move(receiver_desc));
    }

    return desc;
}

// Row tiles whose source cores all sit inside the output rectangle: every source core multicasts its shard (a
// contiguous run of the replica) straight to the whole rectangle at its offset, half on each RISC / NoC. No hub,
// no gather, no semaphore; the program ends when every multicast has landed.
bool can_mcast_rows_directly(const AllGatherForMatmulParams& operation_attributes, const Tensor& input) {
    if (!all_gather_for_matmul_row_tiles(input)) {
        return false;
    }
    const CoreRange output_bbox = operation_attributes.output_core_range_set.bounding_box();
    for (const auto& core : corerange_to_cores(input.shard_spec()->grid, std::nullopt, true)) {
        if (!output_bbox.contains(core)) {
            return false;
        }
    }
    return true;
}

ProgramDescriptor create_rows_mcast_descriptor(
    const AllGatherForMatmulParams& operation_attributes, const AllGatherForMatmulInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    auto* device = input.device();
    const auto& input_shard = input.shard_spec().value();
    const auto input_cores =
        corerange_to_cores(input_shard.grid, std::nullopt, input_shard.orientation == ShardOrientation::ROW_MAJOR);
    const CoreRange output_bbox = operation_attributes.output_core_range_set.bounding_box();
    const CoreCoord start = device->worker_core_from_logical_core(output_bbox.start_coord);
    const CoreCoord end = device->worker_core_from_logical_core(output_bbox.end_coord);
    const uint32_t bbox_num_cores = output_bbox.grid_size().x * output_bbox.grid_size().y;

    const auto& tile = input.tensor_spec().tile();
    const uint32_t tile_bytes = tile.get_tile_size(datatype_to_dataformat_converter(input.dtype()));
    const uint32_t shard_tiles =
        (input_shard.shape[0] / tile.get_tile_shape()[0]) * (input_shard.shape[1] / tile.get_tile_shape()[1]);
    const uint32_t shard_bytes = shard_tiles * tile_bytes;
    const uint32_t first_half_bytes = (shard_tiles / 2) * tile_bytes;

    ProgramDescriptor desc;
    // Several source cores: concurrent multicasts from all of them onto one rectangle contend badly (measured
    // ~7x slower than one source), so they append their shards on a hub that holds none of its own, and the hub
    // multicasts the replica, half on each RISC / NoC.
    std::optional<CoreCoord> hub;
    if (input_cores.size() > 1) {
        for (const auto& core : corerange_to_cores(operation_attributes.output_core_range_set, std::nullopt, true)) {
            if (!input_shard.grid.contains(core)) {
                hub = core;
                break;
            }
        }
    }
    if (hub.has_value()) {
        const CoreCoord hub_physical = device->worker_core_from_logical_core(*hub);
        const CoreRangeSet hub_set{CoreRange(*hub)};
        constexpr uint32_t cb_in = tt::CBIndex::c_0;
        constexpr uint32_t gather_semaphore_id = 0;
        const uint32_t full_bytes = shard_bytes * static_cast<uint32_t>(input_cores.size());
        desc.cbs.push_back(CBDescriptor{
            .total_size = shard_bytes,
            .core_ranges = input_shard.grid,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_in),
                .data_format = datatype_to_dataformat_converter(input.dtype()),
                .page_size = shard_bytes,
            }}},
            .buffer = input.buffer(),
        });
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = gather_semaphore_id,
            .core_ranges = hub_set,
            .initial_value = 0,
        });
        KernelDescriptor sender_desc;
        sender_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek/all_gather_for_matmul/device/kernels/dataflow/"
            "writer_all_gather_for_matmul.cpp";
        sender_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        sender_desc.core_ranges = input_shard.grid;
        sender_desc.compile_time_args = {
            cb_in,
            tt::CBIndex::c_1,
            0u,  // tiles_per_core: no untilize
            1u,  // skip_untilize
            1u,  // num_slices
            1u,  // rows_per_slice
            1u,  // padded_rows_per_slice
            shard_bytes,
            full_bytes,
            gather_semaphore_id,
        };
        sender_desc.config = WriterConfigDescriptor{};
        for (uint32_t i = 0; i < input_cores.size(); ++i) {
            sender_desc.emplace_runtime_args(
                input_cores[i],
                {i, output.buffer(), static_cast<uint32_t>(hub_physical.x), static_cast<uint32_t>(hub_physical.y)});
        }
        KernelDescriptor hub_writer;
        hub_writer.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek/all_gather_for_matmul/device/kernels/dataflow/"
            "hub_mcast_rows.cpp";
        hub_writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
        hub_writer.core_ranges = hub_set;
        hub_writer.compile_time_args = {
            static_cast<uint32_t>(input_cores.size()),
            bbox_num_cores - 1,
            static_cast<uint32_t>(start.x),
            static_cast<uint32_t>(start.y),
            static_cast<uint32_t>(end.x),
            static_cast<uint32_t>(end.y),
            gather_semaphore_id,
        };
        hub_writer.config = WriterConfigDescriptor{};
        KernelDescriptor hub_reader = hub_writer;
        hub_reader.config = ReaderConfigDescriptor{};
        const uint32_t half = (full_bytes / tile_bytes / 2) * tile_bytes;
        hub_writer.emplace_runtime_args(*hub, {output.buffer(), 0u, half});
        hub_reader.emplace_runtime_args(*hub, {output.buffer(), half, full_bytes - half});
        desc.kernels.push_back(std::move(sender_desc));
        desc.kernels.push_back(std::move(hub_writer));
        desc.kernels.push_back(std::move(hub_reader));
        return desc;
    }

    const std::vector<uint32_t> ct_args = {
        static_cast<uint32_t>(start.x),
        static_cast<uint32_t>(start.y),
        static_cast<uint32_t>(end.x),
        static_cast<uint32_t>(end.y),
        bbox_num_cores,
    };
    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek/all_gather_for_matmul/device/kernels/dataflow/"
        "writer_mcast_rows.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = input_shard.grid;
    writer_desc.compile_time_args = ct_args;
    writer_desc.config = WriterConfigDescriptor{};
    KernelDescriptor reader_desc = writer_desc;
    reader_desc.config = ReaderConfigDescriptor{};
    for (uint32_t i = 0; i < input_cores.size(); ++i) {
        const uint32_t dst_offset = i * shard_bytes;
        writer_desc.emplace_runtime_args(
            input_cores[i], {input.buffer(), output.buffer(), 0u, dst_offset, first_half_bytes});
        reader_desc.emplace_runtime_args(
            input_cores[i],
            {input.buffer(),
             output.buffer(),
             first_half_bytes,
             dst_offset + first_half_bytes,
             shard_bytes - first_half_bytes});
    }
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(reader_desc));
    return desc;
}

}  // namespace

ProgramDescriptor AllGatherForMatmulProgramFactory::create_descriptor(
    const AllGatherForMatmulParams& operation_attributes, const AllGatherForMatmulInputs& tensor_args, Tensor& output) {
    if (can_mcast_rows_directly(operation_attributes, tensor_args.input)) {
        return create_rows_mcast_descriptor(operation_attributes, tensor_args, output);
    }
    if (tensor_args.input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
        tensor_args.input.shard_spec()->grid.num_cores() == 1) {
        return create_mcast_descriptor(operation_attributes, tensor_args, output);
    }
    return create_gather_descriptor(operation_attributes, tensor_args, output);
}

}  // namespace ttnn::prim
