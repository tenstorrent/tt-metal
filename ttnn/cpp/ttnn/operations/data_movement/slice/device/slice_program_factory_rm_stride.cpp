// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm_stride.hpp"

#include <optional>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::slice::program {

SliceRmStrideProgramFactory::cached_program_t SliceRmStrideProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, Tensor& output) {
    const auto& input_tensor = tensor_args.input;
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    tt::tt_metal::IDevice* device = input_tensor.device();

    const auto& input_shape = input_tensor.padded_shape();
    const auto& output_shape = output.padded_shape();
    uint32_t element_size = input_tensor.element_size();

    // Calculate total output rows based on tensor rank
    uint32_t total_output_rows = output_shape.volume() / output_shape[-1];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), total_output_rows)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_output_rows);

    // Select kernels based on tensor rank
    std::string reader_kernel_path, writer_kernel_path;
    if (input_shape.rank() <= 4) {
        reader_kernel_path =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_multicore_slice_4d.cpp";
        writer_kernel_path =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/writer_multicore_slice_4d.cpp";
    } else {
        reader_kernel_path =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_multicore_slice_nd.cpp";
        writer_kernel_path =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/writer_multicore_slice_nd.cpp";
    }

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t actual_output_w = output_shape[-1];
    uint32_t output_bytes_per_row = actual_output_w * element_size;
    uint32_t cb_page_size = output_bytes_per_row;

    auto src_buffer_alignment = input_tensor.buffer()->alignment();
    auto dst_buffer_alignment = output.buffer()->alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);

    uint32_t cb_page_size_aligned = tt::round_up(cb_page_size, alignment);
    uint32_t cb_total_size = 2 * cb_page_size_aligned;

    constexpr uint32_t in_cb = 0;

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_total_size, {{in_cb, cb_data_format}})
            .set_page_size(in_cb, cb_page_size_aligned);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    std::vector<uint32_t> reader_compile_time_args = {in_cb, element_size};
    TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {in_cb, element_size};
    TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program, reader_kernel_path, all_cores, tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program, writer_kernel_path, all_cores, tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Calculate runtime arguments helper function
    auto get_slice_runtime_args = [&](const Tensor& input_tensor,
                                      Tensor& output_tensor,
                                      const ttnn::Shape& slice_start,
                                      const ttnn::Shape& slice_end,
                                      const ttnn::Shape& slice_step,
                                      uint32_t num_cores,
                                      uint32_t total_output_rows,
                                      const std::string& reader_kernel_path) {
        const auto& input_shape = input_tensor.padded_shape();
        const auto& output_shape = output_tensor.padded_shape();
        uint32_t element_size = input_tensor.element_size();
        uint32_t tensor_rank = input_shape.rank();

        uint32_t base_rows_per_core = total_output_rows / num_cores;
        uint32_t extra_rows = total_output_rows % num_cores;

        std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores);

        uint32_t row_start_id = 0;
        uint32_t extra_rows_remaining = extra_rows;

        bool using_4d_kernels = (reader_kernel_path.find("4d") != std::string::npos);

        for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
            uint32_t rows_for_this_core = base_rows_per_core;
            if (extra_rows_remaining > 0) {
                rows_for_this_core += 1;
                extra_rows_remaining -= 1;
            }

            std::vector<uint32_t> reader_args, writer_args;

            if (using_4d_kernels) {
                reader_args = {
                    input_tensor.buffer()->address(),
                    tensor_rank,
                    input_shape[-1],
                    input_shape[-2],
                    input_shape[-3],
                    input_shape[-4],
                    output_shape[-1],
                    output_shape[-2],
                    output_shape[-3],
                    output_shape[-4],
                    slice_start[-1],
                    slice_end[-1],
                    slice_step[-1],
                    slice_start[-2],
                    slice_end[-2],
                    slice_step[-2],
                    slice_start[-3],
                    slice_end[-3],
                    slice_step[-3],
                    slice_start[-4],
                    slice_end[-4],
                    slice_step[-4],
                    element_size,
                    rows_for_this_core,
                    row_start_id};

                writer_args = {
                    output_tensor.buffer()->address(),
                    tensor_rank,
                    output_shape[-1],
                    output_shape[-2],
                    output_shape[-3],
                    output_shape[-4],
                    element_size,
                    rows_for_this_core,
                    row_start_id};
            } else {
                reader_args = {
                    input_tensor.buffer()->address(), tensor_rank, element_size, rows_for_this_core, row_start_id};
                reader_args.insert(reader_args.end(), input_shape.cbegin(), input_shape.cend());
                reader_args.insert(reader_args.end(), output_shape.cbegin(), output_shape.cend());
                reader_args.insert(reader_args.end(), slice_start.cbegin(), slice_start.cend());
                reader_args.insert(reader_args.end(), slice_end.cbegin(), slice_end.cend());
                reader_args.insert(reader_args.end(), slice_step.cbegin(), slice_step.cend());

                writer_args = {
                    output_tensor.buffer()->address(), tensor_rank, element_size, rows_for_this_core, row_start_id};
                writer_args.insert(writer_args.end(), output_shape.cbegin(), output_shape.cend());
            }

            ret_val[core_idx] = {reader_args, writer_args};
            row_start_id += rows_for_this_core;
        }

        return ret_val;
    };

    auto all_runtime_args = get_slice_runtime_args(
        input_tensor,
        output,
        args.slice_start,
        args.slice_end,
        args.step,
        num_cores,
        total_output_rows,
        reader_kernel_path);

    auto all_cores_vec = corerange_to_cores(all_cores);
    for (size_t i = 0; i < all_cores_vec.size(); ++i) {
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, all_cores_vec[i], all_runtime_args[i].first);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, all_cores_vec[i], all_runtime_args[i].second);
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, all_cores_vec}};
}

void SliceRmStrideProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*args*/,
    const tensor_args_t& tensor_args,
    Tensor& output) {
    const auto& src_tensor = tensor_args.input;
    const auto& dst_tensor = output;
    const auto& program = cached_program.program;
    const auto& all_cores_vec = cached_program.shared_variables.all_cores_vec;

    for (size_t i = 0; i < cached_program.shared_variables.all_cores_vec.size(); ++i) {
        auto& reader_runtime_args =
            tt::tt_metal::GetRuntimeArgs(program, cached_program.shared_variables.reader_kernel_id, all_cores_vec[i]);
        reader_runtime_args[0] = src_tensor.buffer()->address();

        auto& writer_runtime_args =
            tt::tt_metal::GetRuntimeArgs(program, cached_program.shared_variables.writer_kernel_id, all_cores_vec[i]);
        writer_runtime_args[0] = dst_tensor.buffer()->address();
    }
}

}  // namespace ttnn::operations::data_movement::slice::program
