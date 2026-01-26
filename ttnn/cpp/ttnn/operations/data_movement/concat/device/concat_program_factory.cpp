// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/concat/device/concat_program_factory.hpp"

#include <algorithm>

#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::prim {

ConcatProgramFactory::cached_program_t ConcatProgramFactory::create(
    const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;

    const auto& input_tensors = tensor_args.input_tensors;
    const uint32_t dim = operation_attributes.dim;
    const Tensor& output = tensor_return_value;

    Program program = CreateProgram();
    KernelHandle reader_kernel_id = 0;
    KernelHandle writer_kernel_id = 0;
    std::vector<CoreCoord> cores;

    IDevice* device = output.device();

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output.dtype());

    const bool rm_layout = output.layout() == Layout::ROW_MAJOR;

    constexpr bool rm_orientation = false;

    uint32_t num_output_pages;
    uint32_t single_page_size;
    const uint32_t common_align_len = std::max(input_tensors[0].buffer()->alignment(), output.buffer()->alignment());
    if (rm_layout) {
        num_output_pages = output.physical_volume() / output.padded_shape()[-1];
        single_page_size = tt::align(output.element_size() * output.padded_shape()[-1], common_align_len);
    } else {
        num_output_pages = output.physical_volume() / TILE_HW;
        single_page_size = tt::tile_size(cb_data_format);
    }

    const CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_output_pages, rm_orientation);

    const uint32_t num_input_tensors = input_tensors.size();

    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const uint32_t src0_cb_index = 0;
    const uint32_t num_input_pages = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_pages * single_page_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_page_size);
    CreateCircularBuffer(program, all_cores, cb_src0_config);

    const uint32_t num_dims = output.padded_shape().rank();

    std::vector<uint32_t> src_addr(num_input_tensors);
    std::vector<uint32_t> num_pages_per_block(num_input_tensors);
    std::vector<uint32_t> page_id_per_tensor(num_input_tensors);
    std::vector<uint32_t> page_size_per_tensor(num_input_tensors);

    uint32_t num_accum_pages = 1;
    uint32_t scale_factor = 1;

    // RM is special cased in the loop (dim_units = 1 for last dim else it's the dim size)
    if (!rm_layout) {
        if (dim == num_dims - 2) {
            scale_factor = TILE_HEIGHT;
        } else if (dim == num_dims - 1) {
            scale_factor = TILE_WIDTH;
        }
    }

    for (uint32_t i = dim + 1; i < num_dims; ++i) {
        num_accum_pages *= output.padded_shape()[i];
    }
    if (rm_layout) {
        if (num_dims > 1 && dim < num_dims - 1) {
            num_accum_pages /= output.padded_shape()[-1];
        }
    } else {
        if (dim < num_dims - 2) {
            num_accum_pages /= TILE_HW;
        } else if (dim == num_dims - 2) {
            num_accum_pages /= TILE_WIDTH;
        }
    }

    uint32_t num_output_pages_per_block = 0;

    if (rm_layout) {
        for (uint32_t i = 0; i < num_input_tensors; ++i) {
            auto* buffer = input_tensors[i].buffer();
            src_addr[i] = buffer->address();
            page_size_per_tensor[i] = buffer->page_size();
            if (dim == num_dims - 1) {
                num_pages_per_block[i] = num_accum_pages;
            } else {
                uint32_t dim_pages = input_tensors[i].padded_shape()[dim];
                num_pages_per_block[i] = num_accum_pages * dim_pages;
                num_output_pages_per_block += num_accum_pages * dim_pages;
            }
        }
        if (dim == num_dims - 1) {
            num_output_pages_per_block = 1;
        }
    } else {
        for (uint32_t i = 0; i < num_input_tensors; ++i) {
            auto* buffer = input_tensors[i].buffer();
            src_addr[i] = buffer->address();
            page_size_per_tensor[i] = buffer->page_size();
            uint32_t dim_pages = input_tensors[i].padded_shape()[dim] / scale_factor;
            num_pages_per_block[i] = num_accum_pages * dim_pages;
            num_output_pages_per_block += num_accum_pages * dim_pages;
        }
    }
    std::vector<uint32_t> common_reader_kernel_args = {0, 0, 0};
    common_reader_kernel_args.insert(common_reader_kernel_args.end(), src_addr.cbegin(), src_addr.cend());
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_pages_per_block.cbegin(), num_pages_per_block.cend());

    // Reader compile-time args
    // Data is 32 byte aligned
    std::vector<uint32_t> reader_compile_time_args = {src0_cb_index, num_input_tensors};
    reader_compile_time_args.insert(
        reader_compile_time_args.end(), page_size_per_tensor.cbegin(), page_size_per_tensor.cend());
    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        TensorAccessorArgs(*input_tensors[i].buffer()).append_to(reader_compile_time_args);
    }

    std::map<std::string, std::string> concat_defines;

    if (rm_layout && dim == num_dims - 1) {
        concat_defines["WIDTH_CONCAT"] = "1";
    }

    std::vector<uint32_t> writer_compile_time_args;
    if (rm_layout) {
        writer_compile_time_args = {(std::uint32_t)src0_cb_index, dst_buffer->page_size()};
    } else {
        writer_compile_time_args = {(std::uint32_t)src0_cb_index};
    }
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // Tilized reader
    reader_kernel_id = CreateKernel(
        program,
        rm_layout ? "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
                    "reader_concat_stick_layout_interleaved_start_id.cpp"
                  : "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
                    "reader_concat_interleaved_start_id.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args, concat_defines));

    writer_kernel_id = CreateKernel(
        program,
        rm_layout
            ? "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp"
            : "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, rm_orientation);
    const uint32_t g1_num_cores = core_group_1.num_cores();
    for (uint32_t i = 0, num_pages_written = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        const uint32_t num_pages_per_core =
            (i < g1_num_cores) ? num_tiles_per_core_group_1 : num_tiles_per_core_group_2;
        const uint32_t block_id = num_pages_written / num_output_pages_per_block;
        uint32_t id_within_block = num_pages_written % num_output_pages_per_block;
        uint32_t curr_tensor = 0;
        uint32_t curr_tensor_id = 0;
        for (uint32_t j = 0; j < num_input_tensors; j++) {
            page_id_per_tensor[j] = block_id * num_pages_per_block[j];
            if (id_within_block == 0) {
                continue;
            }
            if (id_within_block >= num_pages_per_block[j]) {
                page_id_per_tensor[j] += num_pages_per_block[j];
                id_within_block -= num_pages_per_block[j];
                curr_tensor = j + 1;
            } else {
                page_id_per_tensor[j] += id_within_block;
                curr_tensor = j;
                curr_tensor_id = id_within_block;
                id_within_block = 0;
            }
        }

        std::vector<uint32_t> reader_kernel_args = common_reader_kernel_args;
        reader_kernel_args[0] = num_pages_per_core;
        reader_kernel_args[1] = curr_tensor;
        reader_kernel_args[2] = curr_tensor_id;
        reader_kernel_args.insert(reader_kernel_args.end(), page_id_per_tensor.begin(), page_id_per_tensor.end());

        std::vector<uint32_t> writer_kernel_args;
        if (rm_layout) {
            writer_kernel_args = {
                dst_buffer->address(), output.buffer()->page_size(), num_pages_per_core, num_pages_written};
        } else {
            writer_kernel_args = {dst_buffer->address(), num_pages_per_core, num_pages_written};
        }
        SetRuntimeArgs(program, reader_kernel_id, core, reader_kernel_args);

        SetRuntimeArgs(program, writer_kernel_id, core, writer_kernel_args);
        num_pages_written += num_pages_per_core;
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, cores}};
}

void ConcatProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ConcatParams& /*operation_attributes*/,
    const ConcatInputs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    const auto& shared_vars = cached_program.shared_variables;

    std::vector<uint32_t> src_addrs(tensor_args.input_tensors.size());
    for (uint32_t i = 0; i < tensor_args.input_tensors.size(); ++i) {
        src_addrs[i] = tensor_args.input_tensors[i].buffer()->address();
    }

    Buffer* dst_buffer = tensor_return_value.buffer();

    for (const CoreCoord& core : shared_vars.cores) {
        {
            auto& runtime_args = GetRuntimeArgs(program, shared_vars.reader_kernel_id, core);
            std::copy(src_addrs.cbegin(), src_addrs.cend(), runtime_args.data() + 3);
        }
        {
            auto& runtime_args = GetRuntimeArgs(program, shared_vars.writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::prim
