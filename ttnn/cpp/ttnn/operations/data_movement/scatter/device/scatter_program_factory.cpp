// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_program_factory.hpp"

#include "scatter_common.hpp"

#include "scatter_device_operation_types.hpp"
#include "tt-metalium/allocator.hpp"
#include "tt-metalium/device.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::data_movement::scatter {

using namespace tt;
using namespace tt::tt_metal;

ScatterProgramFactory::cached_program_t ScatterProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, Tensor& output_tensor) {
    using namespace tt::tt_metal;

    Program program{};

    const auto& input_tensor{tensor_args.input_tensor};
    const auto& input_shape{input_tensor.logical_shape()};
    const auto& index_tensor{tensor_args.index_tensor};
    const auto& index_shape{index_tensor.logical_shape()};
    const auto& src_tensor{tensor_args.src_tensor};
    const auto& src_shape{src_tensor.logical_shape()};
    const auto& output_shape{output_tensor.logical_shape()};

    auto* input_buffer = input_tensor.buffer();
    auto* index_buffer = index_tensor.buffer();
    auto* src_buffer = src_tensor.buffer();
    auto* output_buffer = output_tensor.buffer();

    const uint32_t& input_stick_size = input_shape[-1];
    const uint32_t& index_stick_size = index_shape[-1];
    const uint32_t& source_stick_size = src_shape[-1];
    const uint32_t& output_stick_size = output_shape[-1];

    // input dtype byte sizes
    const uint32_t& input_datum_size = input_tensor.element_size();
    const uint32_t& index_datum_size = index_tensor.element_size();
    const uint32_t& source_datum_size = src_tensor.element_size();
    const uint32_t& output_datum_size = output_tensor.element_size();

    // input row byte sizes
    const uint32_t& input_stick_size_bytes = input_stick_size * input_datum_size;
    const uint32_t& index_stick_size_bytes = index_stick_size * index_datum_size;
    const uint32_t& source_stick_size_bytes = source_stick_size * source_datum_size;
    const uint32_t& output_stick_size_bytes = output_stick_size * output_datum_size;

    // maximal input/index/source/output chunk size, divisible by 32, calculated as follows:
    // BH available L1 mem size of nearly 1.5 MB...
    // ... minimized by the amount of memory reserved by a model...
    // ... divided by 4 to be able to allocate four equally long row chunks (coming from input/index/source/output
    // tensors)
    // ... divided by 4 to account for 4-byte datum sizes of each tensor (fp32, int32)
    // ... minimized by ~10% to account for reserved memory
    const uint32_t input_and_output_max_chunk_size = calculate_optimal_chunk_size(input_tensor);
    const uint32_t index_and_source_max_chunk_size = calculate_optimal_chunk_size(index_tensor);
    const uint32_t input_and_output_chunk_size = std::min(input_stick_size, input_and_output_max_chunk_size);
    const uint32_t index_chunk_size = std::min(index_stick_size, index_and_source_max_chunk_size);
    const uint32_t source_chunk_size = std::min(source_stick_size, index_and_source_max_chunk_size);
    const uint32_t input_and_output_chunk_size_bytes = input_and_output_chunk_size * input_datum_size;
    const uint32_t index_chunk_size_bytes = index_chunk_size * index_datum_size;
    const uint32_t source_chunk_size_bytes = source_chunk_size * source_datum_size;

    // pad pages to 32
    const uint32_t input_page_size_bytes = ceil32(input_and_output_chunk_size_bytes);
    const uint32_t index_page_size_bytes = ceil32(index_chunk_size_bytes);
    const uint32_t source_page_size_bytes = ceil32(source_chunk_size_bytes);
    const uint32_t output_page_size_bytes = ceil32(input_and_output_chunk_size_bytes);

    constexpr const char* reader_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/scatter/device/kernels/dataflow/reader_scatter.cpp";
    constexpr const char* writer_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/scatter/device/kernels/dataflow/writer_scatter.cpp";

    std::vector<uint32_t> compile_time_args{
        input_tensor.buffer()->address(),
        index_tensor.buffer()->address(),
        src_tensor.buffer()->address(),
        output_tensor.buffer()->address(),
        static_cast<uint32_t>(ScatterCB::INPUT),
        static_cast<uint32_t>(ScatterCB::INDEX),
        static_cast<uint32_t>(ScatterCB::SRC),
        static_cast<uint32_t>(ScatterCB::DST),
        input_stick_size,
        index_stick_size,
        source_stick_size,
        output_stick_size,
        input_stick_size_bytes,
        index_stick_size_bytes,
        source_stick_size_bytes,
        output_stick_size_bytes,
        input_shape.rank()};
    tt::tt_metal::TensorAccessorArgs(*input_buffer).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*index_buffer).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*output_buffer).append_to(compile_time_args);

    auto* device = input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t work_units = input_tensor.logical_volume() / input_stick_size;
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
            args.sub_core_grid.has_value()
                ? tt::tt_metal::split_work_to_cores(*args.sub_core_grid, work_units)
                : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, work_units);

    const auto farthest_x_y =
        args.sub_core_grid.has_value() ? args.sub_core_grid->bounding_box().end_coord : compute_with_storage_grid_size;
    const uint32_t all_cores_in_bounding_box = (farthest_x_y.x + 1) * (farthest_x_y.y + 1);
    create_cb(program, input_tensor.dtype(), ScatterCB::INPUT, all_cores, input_page_size_bytes);
    create_cb(program, index_tensor.dtype(), ScatterCB::INDEX, all_cores, index_page_size_bytes);
    create_cb(program, src_tensor.dtype(), ScatterCB::SRC, all_cores, source_page_size_bytes);
    create_cb(program, output_tensor.dtype(), ScatterCB::DST, all_cores, output_page_size_bytes);

    auto reader_kernel =
        create_kernel(program, reader_kernel_path, all_cores, ReaderDataMovementConfig{compile_time_args});
    auto writer_kernel =
        create_kernel(program, writer_kernel_path, all_cores, WriterDataMovementConfig{compile_time_args});

    std::vector<CoreCoord> cores{};
    uint32_t stick_offset = 0;
    for (uint32_t i = 0; i < all_cores_in_bounding_box; ++i) {
        const CoreCoord core{i / (farthest_x_y.y + 1), i % (farthest_x_y.y + 1)};
        uint32_t sticks_per_core;
        if (core_group_1.contains(core)) {
            sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            sticks_per_core = num_sticks_per_core_group_2;
        } else {
            continue;
        }
        cores.push_back(core);

        std::vector<uint32_t> reader_runtime_args{
            input_buffer->address(),
            index_buffer->address(),
            src_buffer->address(),
            stick_offset,
            sticks_per_core,
            input_and_output_chunk_size,
            index_chunk_size,
            source_chunk_size,
            static_cast<uint32_t>(args.opt_reduction)};
        std::copy(input_shape.cbegin(), input_shape.cend() - 1, std::back_inserter(reader_runtime_args));
        std::copy(index_shape.cbegin(), index_shape.cend() - 1, std::back_inserter(reader_runtime_args));

        SetRuntimeArgs(program, reader_kernel, core, reader_runtime_args);

        std::vector<uint32_t> writer_runtime_args{
            output_buffer->address(),
            stick_offset,
            sticks_per_core,
            input_and_output_chunk_size,
        };

        SetRuntimeArgs(program, writer_kernel, core, writer_runtime_args);

        stick_offset += sticks_per_core;
    }

    return {std::move(program), {reader_kernel, writer_kernel, cores}};
}

void ScatterProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*args*/,
    const tensor_args_t& tensor_args,
    Tensor& output_tensor) {
    const auto& program = cached_program.program;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& cores = cached_program.shared_variables.cores;

    auto input_buffer_address = tensor_args.input_tensor.buffer()->address();
    auto index_buffer_address = tensor_args.index_tensor.buffer()->address();
    auto source_buffer_address = tensor_args.src_tensor.buffer()->address();
    auto output_buffer_address = output_tensor.buffer()->address();
    for (const auto& core : cores) {
        auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
        auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
        reader_runtime_args[0] = input_buffer_address;
        reader_runtime_args[1] = index_buffer_address;
        reader_runtime_args[2] = source_buffer_address;
        writer_runtime_args[0] = output_buffer_address;
    }
}

}  // namespace ttnn::operations::data_movement::scatter
