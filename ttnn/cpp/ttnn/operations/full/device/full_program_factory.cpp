// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "full_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/cb_utils.hpp"

using namespace tt;
using namespace tt::constants;

union datatype {
    uint32_t u32;
    float f32;
} u;
namespace ttnn::operations::full {
FullOperation::ProgramFactory::cached_program_t FullOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    [[maybe_unused]] const tensor_args_t&,
    tensor_return_value_t& output) {
    auto dtype = operation_attributes.dtype;
    auto fill_value = operation_attributes.fill_value;

    auto grid = operation_attributes.mesh_device->compute_with_storage_grid_size();
    auto num_pages = (uint32_t)output.buffer()->num_pages();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(grid, num_pages);

    uint32_t page_size = output.buffer()->page_size();
    TT_FATAL(page_size % output.element_size() == 0, "Page size must be divisible by element size");
    uint32_t elems_per_page = page_size / output.element_size();

    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(dtype);

    // Create program
    Program program = Program();

    // Create circular buffer
    auto cb_index = tt::CBIndex::c_0;
    tt::tt_metal::create_cb(cb_index, program, all_cores, page_size, 1, data_format);

    // Create kernels
    std::map<std::string, std::string> reader_defines;
    switch (dtype) {
        case DataType::BFLOAT16: reader_defines["OUTPUT_DTYPE_BFLOAT16"] = "1"; break;
        case DataType::INT32: reader_defines["OUTPUT_DTYPE_INT32"] = "1"; break;
        case DataType::FLOAT32: reader_defines["OUTPUT_DTYPE_FLOAT32"] = "1"; break;
        default: break;
    }

    if (std::holds_alternative<int>(fill_value)) {
        u.u32 = std::get<int>(fill_value);
    } else if (std::holds_alternative<float>(fill_value)) {
        auto float_fill_value = std::get<float>(fill_value);
        if (dtype == DataType::BFLOAT16) {
            u.u32 = static_cast<uint32_t>(std::bit_cast<uint16_t>(bfloat16(float_fill_value))) << 16;
        } else {
            u.f32 = float_fill_value;
        }
    }

    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)cb_index, elems_per_page, page_size};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    auto writer_id = CreateWriteKernel(
        program,
        "ttnn/cpp/ttnn/operations/full/device/kernels/writer_full.cpp",
        all_cores,
        writer_compile_time_args,
        reader_defines);

    auto cores = corerange_to_cores(all_cores, std::nullopt);

    // If there are more pages than cores, we use NCRISC to split the work
    std::optional<tt::tt_metal::KernelHandle> reader_id = std::nullopt;
    if (num_pages > num_cores) {
        // Create a second circular buffer for the reader
        auto cb_index2 = tt::CBIndex::c_1;
        tt::tt_metal::create_cb(cb_index2, program, all_cores, page_size, 1, data_format);

        // Create the reader compile time arguments
        std::vector<uint32_t> reader_compile_time_args = {(uint32_t)cb_index2, elems_per_page, page_size};
        tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(reader_compile_time_args);

        // Create the reader kernel
        reader_id = CreateReadKernel(
            program,
            "ttnn/cpp/ttnn/operations/full/device/kernels/writer_full.cpp",
            all_cores,
            reader_compile_time_args,
            reader_defines);
    }

    // Set runtime arguments
    uint32_t page_offset = 0;

    for (const auto& core : cores) {
        uint32_t num_pages_per_core;
        if (core_group_1.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        if (reader_id.has_value()) {
            uint32_t reader_page_start = page_offset;
            uint32_t num_pages_per_reader = num_pages_per_core / 2;
            std::vector<uint32_t> reader_args = {
                output.buffer()->address(), u.u32, num_pages_per_reader, reader_page_start};
            SetRuntimeArgs(program, reader_id.value(), core, reader_args);

            uint32_t writer_page_start = reader_page_start + num_pages_per_reader;
            uint32_t num_pages_per_writer = num_pages_per_core - num_pages_per_reader;
            std::vector<uint32_t> writer_args = {
                output.buffer()->address(), u.u32, num_pages_per_writer, writer_page_start};
            SetRuntimeArgs(program, writer_id, core, writer_args);
        } else {
            std::vector<uint32_t> writer_args = {output.buffer()->address(), u.u32, num_pages_per_core, page_offset};
            SetRuntimeArgs(program, writer_id, core, writer_args);
        }
        page_offset += num_pages_per_core;
    }
    return {std::move(program), {writer_id, reader_id, cores}};
}

void FullOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    [[maybe_unused]] const tensor_args_t&,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& writer_kernel_id = cached_program.shared_variables.writer_id;
    auto& reader_kernel_id = cached_program.shared_variables.reader_id;
    auto& cores = cached_program.shared_variables.cores;
    for (const auto& core : cores) {
        if (reader_kernel_id.has_value()) {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id.value(), core);
            runtime_args[0] = output.buffer()->address();
        }
        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = output.buffer()->address();
        }
    }
}
}  // namespace ttnn::operations::full
