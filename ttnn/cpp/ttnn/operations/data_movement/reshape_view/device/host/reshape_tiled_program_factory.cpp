// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/data_movement/reshape_view/reshape_common.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"

#include "ttnn/operations/data_movement/reshape_view/device/reshape_device_operation.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape_common.hpp"
#include "reshape_program_factory.hpp"

namespace ttnn::operations::data_movement::reshape {

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

tt::tt_metal::operation::ProgramWithCallbacks reshape_tiled_program_factory(
    const Tensor& input_tensor, const Tensor& output_tensor) {
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

    auto compressed_map = detail::compute_reshape_map(
        num_input_pages, num_output_pages, input_shape, output_shape, tile_shape, face_shape);

    const auto grid = device->compute_with_storage_grid_size();

    uint32_t num_cores_x = grid.x;
    uint32_t num_cores_y = grid.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // PCC fails when this is greater than 1. TODO figure out why.
    constexpr auto reader_cb_len = 1;

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
            tt::tt_metal::split_work_to_cores(grid, num_output_pages);

    TT_ASSERT(num_cores <= num_output_pages);

    std::vector<uint32_t> reader_compile_time_args = {input_tile_size_bytes, input_cb_idx};
    tt::tt_metal::TensorAccessorArgs(*input_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/dataflow/reader_reshape_tiled.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    std::vector<uint32_t> writer_compile_time_args = {
        input_tile_size_bytes, tt::datum_size(output_cb_data_format), input_cb_idx, output_cb_idx};
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

        // Build per-core rt_args from compressed_map.page_pattern_runs
        std::vector<uint32_t> core_rt_args;
        size_t num_short_runs = 0, num_long_runs = 0;

        // Count runs by type
        std::set<uint32_t> used_template_indices;
        for (const auto& run : compressed_map.page_pattern_runs) {
            if (run.output_page_index_end < page_idx_start || run.output_page_index_start >= page_idx_end) {
                continue;
            }
            used_template_indices.insert(run.pattern_template_index);
            if (run.run_length == 1) {
                num_short_runs++;
            } else {
                num_long_runs++;
            }
        }
        std::vector<uint32_t> global_to_local_template_map(compressed_map.pattern_templates.size(), UINT32_MAX);
        std::vector<uint32_t> core_template_indices;
        uint32_t local_idx = 0;
        for (uint32_t global_idx : used_template_indices) {
            global_to_local_template_map[global_idx] = local_idx++;
            core_template_indices.push_back(global_idx);
        }

        // Pack short runs first, then long runs
        for (const auto& run : compressed_map.page_pattern_runs) {
            if (run.output_page_index_end < page_idx_start || run.output_page_index_start >= page_idx_end) {
                continue;
            }
            uint32_t start = std::max(run.output_page_index_start, page_idx_start);
            uint32_t end = std::min(run.output_page_index_end, page_idx_end - 1);

            uint32_t local_template_idx = global_to_local_template_map[run.pattern_template_index];

            if (run.run_length == 1) {
                core_rt_args.push_back(detail::pack_rt_short(start, end));
                core_rt_args.push_back(detail::pack_rt_short(run.input_page_index_start, local_template_idx));
                core_rt_args.push_back(detail::pack_rt_short(run.input_offset_start, run.output_offset_start));
            }
        }

        for (const auto& run : compressed_map.page_pattern_runs) {
            if (run.output_page_index_end < page_idx_start || run.output_page_index_start >= page_idx_end) {
                continue;
            }
            uint32_t start = std::max(run.output_page_index_start, page_idx_start);
            uint32_t end = std::min(run.output_page_index_end, page_idx_end - 1);
            uint32_t local_template_idx = global_to_local_template_map[run.pattern_template_index];

            if (run.run_length > 1) {
                // Long format: 10 fields
                core_rt_args.push_back(detail::pack_rt_short(start, end));
                core_rt_args.push_back(detail::pack_rt_short(run.input_page_index_start, local_template_idx));
                core_rt_args.push_back(detail::pack_rt_short(run.input_offset_start, run.output_offset_start));
                core_rt_args.push_back(detail::pack_rt_short(run.run_length, run.input_page_index_stride));
                core_rt_args.push_back(detail::pack_rt_short(run.input_offset_stride, run.output_offset_stride));
            }
        }

        // Build final RT args vector
        std::vector<uint32_t> reader_runtime_args;
        reader_runtime_args.push_back(used_template_indices.size());             // num_templates
        reader_runtime_args.push_back(num_short_runs);                           // num_short_runs
        reader_runtime_args.push_back(num_long_runs);                            // num_long_runs
        reader_runtime_args.push_back(input_buffer->address());                  // buffer_addr

        // Add pattern templates
        for (uint32_t global_idx : core_template_indices) {
            const auto& tmpl = compressed_map.pattern_templates[global_idx];
            reader_runtime_args.push_back(tmpl.input_page_stride);
            reader_runtime_args.push_back(tmpl.input_offset_stride);
            reader_runtime_args.push_back(tmpl.output_offset_stride);
            reader_runtime_args.push_back(tmpl.num_elements);
        }
        // Add run data
        for (auto k : core_rt_args) {
            reader_runtime_args.push_back(k);
        }
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, c, reader_runtime_args);

        // Same for writer
        std::vector<uint32_t> writer_runtime_args;
        writer_runtime_args.push_back(used_template_indices.size());
        writer_runtime_args.push_back(num_short_runs);
        writer_runtime_args.push_back(num_long_runs);
        writer_runtime_args.push_back(output_buffer->address());
        for (uint32_t global_idx : core_template_indices) {
            const auto& tmpl = compressed_map.pattern_templates[global_idx];
            writer_runtime_args.push_back(tmpl.input_page_stride);
            writer_runtime_args.push_back(tmpl.input_offset_stride);
            writer_runtime_args.push_back(tmpl.output_offset_stride);
            writer_runtime_args.push_back(tmpl.num_elements);
        }
        for (auto k : core_rt_args) {
            writer_runtime_args.push_back(k);
        }
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, c, writer_runtime_args);
        page_idx_start += increment;
        utilized_cores.push_back(c);
    }

    auto override_runtime_arguments_callback =
        [reader_kernel_id, writer_kernel_id, utilized_cores](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input_tensor = input_tensors.at(0);
            const auto& output_tensor = output_tensors.at(0);

            const auto input_buffer_addr = input_tensor.buffer()->address();
            const auto output_buffer_addr = output_tensor.buffer()->address();

            for (const auto& core : utilized_cores) {
                auto& reader_runtime_args_core = GetRuntimeArgs(program, reader_kernel_id, core);
                reader_runtime_args_core.at(3) = input_buffer_addr;

                auto& writer_runtime_args_core = GetRuntimeArgs(program, writer_kernel_id, core);
                writer_runtime_args_core.at(3) = output_buffer_addr;
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

};  // namespace ttnn::operations::data_movement::reshape
