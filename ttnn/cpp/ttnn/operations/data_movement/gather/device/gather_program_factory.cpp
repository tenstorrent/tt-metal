// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::prim {
using namespace tt::tt_metal;

namespace {

constexpr const char* kReaderSingle =
    "ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_single_row_single_core.cpp";
constexpr const char* kWriterSingle =
    "ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_single_row_single_core.cpp";
constexpr const char* kReaderMulti =
    "ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_single_row_multi_core.cpp";
constexpr const char* kWriterMulti =
    "ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_single_row_multi_core.cpp";

}  // namespace

ProgramDescriptor GatherDeviceOperation::SingleRowSingleCore::create_descriptor(
    const GatherParams& attributes, const GatherInputs& tensor_args, Tensor& output_tensor) {
    const tt::DataFormat input_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    const tt::DataFormat input_index_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_index_tensor.dtype());
    const tt::DataFormat output_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());

    const uint32_t input_tensor_tile_size = tile_size(input_tensor_cb_data_format);
    const uint32_t input_index_tensor_tile_size = tile_size(input_index_tensor_cb_data_format);
    const uint32_t output_tensor_tile_size = tile_size(output_tensor_cb_data_format);

    auto* input_tensor_buffer = tensor_args.input_tensor.buffer();
    auto* input_index_tensor_buffer = tensor_args.input_index_tensor.buffer();
    auto* output_tensor_buffer = output_tensor.buffer();

    const bool input_tensor_is_dram = input_tensor_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool input_index_tensor_is_dram = input_index_tensor_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool output_tensor_is_dram = output_tensor_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    const auto tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const auto tile_height = tensor_args.input_tensor.tensor_spec().tile().get_height();

    const auto input_index_shape = tensor_args.input_index_tensor.padded_shape();
    const auto input_shape = tensor_args.input_tensor.padded_shape();
    const uint32_t Ht = (input_index_shape[0] * input_index_shape[1] * input_index_shape[2]) / tile_height;
    const uint32_t Wt_input = input_shape[3] / tile_width;
    const uint32_t Wt_index = input_index_shape[3] / tile_width;

    // Calculate the number of cores available for computation
    auto* device = tensor_args.input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t max_number_of_cores = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;

    // Create core grid
    CoreRangeSet core_grid =
        tt::tt_metal::num_cores_to_corerangeset(max_number_of_cores, compute_with_storage_grid_size, true);
    // Override core grid if sub_core_grids is provided in operation attributes
    if (attributes.sub_core_grids.has_value()) {
        core_grid = attributes.sub_core_grids.value();
    }

    const auto
        [total_number_of_cores,       // number of cores utilized
         core_range,                  // set of all cores used
         core_group_1,                // Primary core group
         core_group_2,                // Secondary core group
         num_tiles_per_core_group_1,  // Number of tiles each core in the primary group processes
         num_tiles_per_core_group_2   // Number of tiles each core in the secondary group processes
    ] = tt::tt_metal::split_work_to_cores(core_grid, Ht, true);
    const auto work_groups = {
        std::make_pair(core_group_1, num_tiles_per_core_group_1),
        std::make_pair(core_group_2, num_tiles_per_core_group_2)};

    ProgramDescriptor desc;

    constexpr uint32_t input_tensor_cb_index = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = Wt_input * input_tensor_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_tensor_cb_index),
            .data_format = input_tensor_cb_data_format,
            .page_size = input_tensor_tile_size,
        }}},
    });

    constexpr uint32_t input_index_tensor_cb_index = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = input_index_tensor_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_index_tensor_cb_index),
            .data_format = input_index_tensor_cb_data_format,
            .page_size = input_index_tensor_tile_size,
        }}},
    });

    constexpr uint32_t output_tensor_cb_index = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_tensor_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_tensor_cb_index),
            .data_format = output_tensor_cb_data_format,
            .page_size = output_tensor_tile_size,
        }}},
    });

    std::vector<uint32_t> reader_compile_time_args = {
        input_tensor_cb_index,
        input_index_tensor_cb_index,
        output_tensor_cb_index,
        static_cast<uint32_t>(input_index_tensor_is_dram),
        Ht,
        Wt_input,
        Wt_index,
        total_number_of_cores,
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y};
    TensorAccessorArgs(*input_index_tensor_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kReaderSingle;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_range;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_compile_time_args = {
        input_tensor_cb_index,
        output_tensor_cb_index,
        static_cast<uint32_t>(input_tensor_is_dram),
        static_cast<uint32_t>(output_tensor_is_dram),
        Ht,
        Wt_input,
        Wt_index,
        total_number_of_cores,
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y};
    TensorAccessorArgs(*input_tensor_buffer).append_to(writer_compile_time_args);
    TensorAccessorArgs(*output_tensor_buffer).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterSingle;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_range;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};

    uint32_t id = 0;  // Offset for the next core in the group
    for (const auto& [group, work_per_core] : work_groups) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                reader_desc.runtime_args.emplace_back(
                    core,
                    KernelDescriptor::CoreRuntimeArgs{
                        input_index_tensor_buffer->address(), work_per_core, tile_width, tile_height, id});
                writer_desc.runtime_args.emplace_back(
                    core,
                    KernelDescriptor::CoreRuntimeArgs{
                        input_tensor_buffer->address(), output_tensor_buffer->address(), work_per_core, id});
                id++;
            }
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

ProgramDescriptor GatherDeviceOperation::SingleRowMultiCore::create_descriptor(
    const GatherParams& attributes, const GatherInputs& tensor_args, Tensor& output_tensor) {
    const tt::DataFormat input_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    const tt::DataFormat input_index_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_index_tensor.dtype());
    const tt::DataFormat output_tensor_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());

    const uint32_t input_tensor_tile_size = tile_size(input_tensor_cb_data_format);
    const uint32_t input_index_tensor_tile_size = tile_size(input_index_tensor_cb_data_format);
    const uint32_t output_tensor_tile_size = tile_size(output_tensor_cb_data_format);

    auto* const input_tensor_buffer = tensor_args.input_tensor.buffer();
    auto* const input_index_tensor_buffer = tensor_args.input_index_tensor.buffer();
    auto* const output_tensor_buffer = output_tensor.buffer();

    const bool input_tensor_is_dram = input_tensor_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool input_index_tensor_is_dram = input_index_tensor_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool output_tensor_is_dram = output_tensor_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    const auto tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const auto tile_height = tensor_args.input_tensor.tensor_spec().tile().get_height();

    const auto input_index_shape = tensor_args.input_index_tensor.padded_shape();
    const auto input_shape = tensor_args.input_tensor.padded_shape();
    const uint32_t Ht = (input_index_shape[0] * input_index_shape[1] * input_index_shape[2]) / tile_height;
    const uint32_t Wt_input = input_shape[3] / tile_width;
    const uint32_t Wt_index = input_index_shape[3] / tile_width;

    // Calculate the number of cores available for computation
    auto* device = tensor_args.input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t max_number_of_cores = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;

    // Create core grid
    CoreRangeSet core_grid =
        tt::tt_metal::num_cores_to_corerangeset(max_number_of_cores, compute_with_storage_grid_size, true);
    // Override core grid if sub_core_grids is provided in operation attributes
    if (attributes.sub_core_grids.has_value()) {
        core_grid = attributes.sub_core_grids.value();
    }

    const auto
        [total_number_of_cores,       // number of cores utilized
         core_range,                  // set of all cores used
         core_group_1,                // Primary core group
         core_group_2,                // Secondary core group
         num_tiles_per_core_group_1,  // Number of tiles each core in the primary group processes
         num_tiles_per_core_group_2   // Number of tiles each core in the secondary group processes
    ] = tt::tt_metal::split_work_to_cores(core_grid, Wt_index, true);
    const auto work_groups = {
        std::make_pair(core_group_1, num_tiles_per_core_group_1),
        std::make_pair(core_group_2, num_tiles_per_core_group_2)};

    ProgramDescriptor desc;
    constexpr uint32_t buffer_scale_factor = 2;

    constexpr uint32_t input_tensor_cb_index = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = buffer_scale_factor * input_tensor_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_tensor_cb_index),
            .data_format = input_tensor_cb_data_format,
            .page_size = input_tensor_tile_size,
        }}},
    });

    constexpr uint32_t input_index_tensor_cb_index = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = input_index_tensor_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_index_tensor_cb_index),
            .data_format = input_index_tensor_cb_data_format,
            .page_size = input_index_tensor_tile_size,
        }}},
    });

    constexpr uint32_t output_tensor_cb_index = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_tensor_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_tensor_cb_index),
            .data_format = output_tensor_cb_data_format,
            .page_size = output_tensor_tile_size,
        }}},
    });

    std::vector<uint32_t> reader_compile_time_args = {
        input_tensor_cb_index,
        input_index_tensor_cb_index,
        output_tensor_cb_index,
        static_cast<uint32_t>(input_index_tensor_is_dram),
        Ht,
        Wt_input,
        Wt_index,
        total_number_of_cores,
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y};
    TensorAccessorArgs(*input_index_tensor_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kReaderMulti;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_range;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_compile_time_args = {
        input_tensor_cb_index,
        output_tensor_cb_index,
        static_cast<uint32_t>(input_tensor_is_dram),
        static_cast<uint32_t>(output_tensor_is_dram),
        Ht,
        Wt_input,
        Wt_index,
        total_number_of_cores,
        compute_with_storage_grid_size.x,
        compute_with_storage_grid_size.y};
    TensorAccessorArgs(*input_tensor_buffer).append_to(writer_compile_time_args);
    TensorAccessorArgs(*output_tensor_buffer).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterMulti;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_range;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};

    uint32_t id = 0;  // Offset for the next core in the group
    for (const auto& [group, work_per_core] : work_groups) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                reader_desc.runtime_args.emplace_back(
                    core,
                    KernelDescriptor::CoreRuntimeArgs{
                        input_index_tensor_buffer->address(), work_per_core, tile_width, tile_height, id});
                writer_desc.runtime_args.emplace_back(
                    core,
                    KernelDescriptor::CoreRuntimeArgs{
                        input_tensor_buffer->address(), output_tensor_buffer->address(), work_per_core, id});
                id++;
            }
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

}  // namespace ttnn::prim
