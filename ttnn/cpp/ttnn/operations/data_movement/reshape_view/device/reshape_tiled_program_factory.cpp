// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/device/reshape_tiled_program_factory.hpp"

#include <math.h>
#include <numeric>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/data_movement/reshape_view/reshape_common.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/reshape_device_operation.hpp"

namespace ttnn::operations::data_movement::reshape {

namespace detail {

Tensor compute_reshape_mapping_host_tensor(
    const uint32_t num_input_pages,
    const uint32_t num_output_pages,
    const Shape& input_shape,
    const Shape& output_shape,
    const std::array<uint32_t, 2>& tile_shape,
    const std::array<uint32_t, 2>& face_shape) {
    Dims tile_dims_input(input_shape, tile_shape), tile_dims_output(output_shape, tile_shape);

    std::vector<std::vector<SegmentMapData>> mapping_vector;
    mapping_vector.reserve(num_output_pages);

    for (uint32_t output_page_idx = 0; output_page_idx < num_output_pages; ++output_page_idx) {
        mapping_vector.emplace_back(reshape_map_output_page(
            output_page_idx, input_shape, output_shape, tile_dims_input, tile_dims_output, tile_shape, face_shape));
    }

    // flatten again
    uint32_t max_input_segments =
        std::max_element(mapping_vector.begin(), mapping_vector.end(), [](const auto& a, const auto& b) {
            return a.size() < b.size();
        })->size();

    // Ensure that map data is always aligned
    max_input_segments += max_input_segments % (tt::tt_metal::hal::get_l1_alignment());

    // initialize to 0 because that will be checked by the kernel as a stopping condition
    std::vector<uint32_t> flat_mapping_vector(SegmentMapData::size * num_output_pages * max_input_segments, 0);
    auto it = flat_mapping_vector.begin();
    for (const auto& v : mapping_vector) {
        auto* map_ptr = reinterpret_cast<SegmentMapData*>(&(*it));
        std::copy(v.begin(), v.end(), map_ptr);

        it += max_input_segments * SegmentMapData::size;
    }

    const std::array<uint32_t, 2> mapping_shape_vector = {num_output_pages, SegmentMapData::size * max_input_segments};
    const Shape mapping_shape(mapping_shape_vector);
    const tt::tt_metal::TensorLayout mapping_layout(
        tt::tt_metal::convert_to_data_type<decltype(flat_mapping_vector)::value_type>(),
        ttnn::ROW_MAJOR_LAYOUT,
        MemoryConfig());

    return Tensor::from_vector(flat_mapping_vector, TensorSpec(mapping_shape, mapping_layout));
}
}  // namespace detail

// Algorithm overview:
// The host computes the mapping between input shape and the output shapes as a series of data segments that are
// contiguous for both input and output tensors. The mapping data is stored as 4 integers per segment: input page index,
// offset of the segment in the input page, offset of the segment in the output page, number of elements in the segment;
// the ordering of the segments in the map are concomitant with the ordering of the output tensor pages. The mapping
// data is stored as an auxiliary integer tensor where each page corresponds to a page of the output tensor.

// The device operation is parallelized over output tensor pages, where each core operates on a range of pages.

// The reader kernel loads the mapping tensor page that corresponds to the current output tensor page on which it is
// operating and pushes it on to the circular buffer. The reader kernel loops over all of the data segments represented
// by the map and loads the specified input pages, avoiding redundant loads of pages for segments that come from the
// same input page, and pushes them to the circular buffer.

// The writer kernel pops mapping pages off the circular buffer, corresponding to the current page. It loops through
// the input tensor pages specified by the map and, as necessary, pops input pages off the circular buffer, again
// accounting for consecutive segments that come from the same input page. Using the offsets and size supplied by the
// map, the reader copies the segment from the input page to a scratch page stored in L1. When all segments are written,
// the scratch page is copied to its output destination.

ReshapeTiledProgramFactory::cached_program_t ReshapeTiledProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor = tensor_args.input;
    const auto& output_tensor = tensor_return_value;

    const auto& input_shape = input_tensor.logical_shape();
    const auto& output_shape = output_tensor.logical_shape();

    TT_FATAL(input_shape.volume() == output_shape.volume(), "Requested shapes are not of equal volume");

    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const auto& face_shape = input_tensor.tensor_spec().tile().get_face_shape();

    TT_ASSERT(input_shape.size() == 3 && output_shape.size() == 3, "Kernel designed for rank 3 tensors");

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::tt_metal::distributed::MeshDevice* device = input_tensor.device();

    tt::tt_metal::Buffer* input_buffer = input_tensor.buffer();
    tt::tt_metal::Buffer* output_buffer = output_tensor.buffer();

    TT_ASSERT(input_buffer != nullptr, "Output buffer should be allocated on device!");

    const uint32_t num_input_pages = tt::div_up(input_tensor.physical_volume(), tile_shape[0] * tile_shape[1]);
    const uint32_t num_output_pages = tt::div_up(output_tensor.physical_volume(), tile_shape[0] * tile_shape[1]);

    Tensor mapping_tensor = detail::compute_reshape_mapping_host_tensor(
                                num_input_pages, num_output_pages, input_shape, output_shape, tile_shape, face_shape)
                                .to_device(device);

    tt::tt_metal::Buffer* mapping_buffer = mapping_tensor.buffer();
    const auto grid = device->compute_with_storage_grid_size();

    uint32_t num_cores_x = grid.x;
    uint32_t num_cores_y = grid.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // set up CB for mapping metadata

    // PCC fails when this is greater than 1. TODO figure out why.
    constexpr auto reader_cb_len = 1;

    auto mapping_page_size = mapping_tensor.logical_shape()[-1];
    auto mapping_dataformat = tt::tt_metal::datatype_to_dataformat_converter(mapping_tensor.dtype());
    auto mapping_page_size_bytes = mapping_page_size * mapping_tensor.element_size();
    constexpr auto mapping_cb_idx = tt::CBIndex::c_0;

    const tt::tt_metal::CircularBufferConfig cb_mapping_config =
        tt::tt_metal::CircularBufferConfig(
            mapping_page_size_bytes * reader_cb_len, {{mapping_cb_idx, mapping_dataformat}})
            .set_page_size(mapping_cb_idx, mapping_page_size_bytes);
    tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_mapping_config);

    // set up CB for input tiles
    const auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const auto input_tile_size_bytes = tt::tile_size(input_cb_data_format);
    constexpr auto input_cb_idx = tt::CBIndex::c_1;

    tt::tt_metal::CircularBufferConfig cb_input_config =
        tt::tt_metal::CircularBufferConfig(
            input_tile_size_bytes * reader_cb_len, {{input_cb_idx, input_cb_data_format}})
            .set_page_size(input_cb_idx, input_tile_size_bytes);
    tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_input_config);

    // TODO assert output tile size and data format same as input
    const auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    const auto output_tile_size_bytes = tt::tile_size(output_cb_data_format);
    constexpr auto output_cb_idx = tt::CBIndex::c_2;

    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(output_tile_size_bytes, {{output_cb_idx, output_cb_data_format}})
            .set_page_size(output_cb_idx, output_tile_size_bytes);
    tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_output_config);

    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
            operation_attributes.sub_core_grid.has_value()
                ? tt::tt_metal::split_work_to_cores(operation_attributes.sub_core_grid.value(), num_output_pages)
                : tt::tt_metal::split_work_to_cores(grid, num_output_pages);

    TT_ASSERT(num_cores <= num_output_pages);

    std::vector<uint32_t> reader_compile_time_args = {
        mapping_page_size_bytes, input_tile_size_bytes, mapping_cb_idx, input_cb_idx};
    tt::tt_metal::TensorAccessorArgs(*mapping_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*input_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/dataflow/reader_reshape_tiled.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    const uint32_t max_map_entries = mapping_page_size / detail::SegmentMapData::size;
    std::vector<uint32_t> writer_compile_time_args = {
        input_tile_size_bytes,
        max_map_entries,
        tt::datum_size(output_cb_data_format),
        mapping_cb_idx,
        input_cb_idx,
        output_cb_idx};
    tt::tt_metal::TensorAccessorArgs(*output_buffer).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/dataflow/"
        "writer_reshape_tiled.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    uint32_t page_idx_start = 0, page_idx_end = 0;
    std::vector<CoreCoord> utilized_cores;
    for (auto c : corerange_to_cores(all_cores, std::nullopt)) {
        uint32_t increment = 0;
        if (core_group_1.contains(c)) {
            increment = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(c)) {
            increment = num_tiles_per_core_group_2;
        } else {
            continue;
        }
        page_idx_end += increment;

        const std::vector<uint32_t> reader_runtime_args = {
            input_buffer->address(), mapping_buffer->address(), page_idx_start, page_idx_end};
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, c, reader_runtime_args);

        const std::vector<uint32_t> writer_runtime_args = {output_buffer->address(), page_idx_start, page_idx_end};

        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, c, writer_runtime_args);

        page_idx_start += increment;
        utilized_cores.push_back(c);
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, utilized_cores, mapping_tensor}};
}

void ReshapeTiledProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& shared_variables = cached_program.shared_variables;
    const auto& reader_kernel_id = shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = shared_variables.writer_kernel_id;
    const auto& utilized_cores = shared_variables.utilized_cores;
    auto& mapping_tensor = shared_variables.mapping_tensor;

    const auto& input_tensor = tensor_args.input;
    const auto& output_tensor = tensor_return_value;

    if (operation_attributes.recreate_mapping_tensor) {
        const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
        const auto& face_shape = input_tensor.tensor_spec().tile().get_face_shape();
        const uint32_t num_input_pages = tt::div_up(input_tensor.physical_volume(), tile_shape[0] * tile_shape[1]);
        const uint32_t num_output_pages = tt::div_up(output_tensor.physical_volume(), tile_shape[0] * tile_shape[1]);

        mapping_tensor = detail::compute_reshape_mapping_host_tensor(
                             num_input_pages,
                             num_output_pages,
                             input_tensor.logical_shape(),
                             output_tensor.logical_shape(),
                             tile_shape,
                             face_shape)
                             .to_device(input_tensor.device());
    }

    const auto input_buffer_addr = input_tensor.buffer()->address();
    const auto output_buffer_addr = output_tensor.buffer()->address();
    auto& program = cached_program.program;

    for (const auto& core : utilized_cores) {
        auto& reader_runtime_args_core = GetRuntimeArgs(program, reader_kernel_id, core);
        reader_runtime_args_core.at(0) = input_buffer_addr;
        if (operation_attributes.recreate_mapping_tensor) {
            reader_runtime_args_core.at(1) = mapping_tensor.buffer()->address();
        }

        auto& writer_runtime_args_core = GetRuntimeArgs(program, writer_kernel_id, core);
        writer_runtime_args_core.at(0) = output_buffer_addr;
    }
}

}  // namespace ttnn::operations::data_movement::reshape
