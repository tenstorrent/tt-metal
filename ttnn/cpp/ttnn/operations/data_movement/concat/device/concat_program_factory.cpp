// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/concat/device/concat_program_factory.hpp"

#include <algorithm>

#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/hal.hpp>

namespace ttnn::prim {

ConcatProgramFactory::cached_program_t ConcatProgramFactory::create(
    const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;

    const auto& input_tensors = tensor_args.input_tensors;
    const uint32_t dim = operation_attributes.dim;
    const Tensor& output = tensor_return_value;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

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

    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_cores;
    uint32_t num_tiles_per_core_group_1;
    uint32_t num_tiles_per_core_group_2;
    uint32_t num_cores_x = 0;
    uint32_t num_cores_y = 0;
    std::vector<CoreCoord> cores_list;

    if (sub_core_grids.has_value() && !output.is_sharded()) {
        // Use sub_core_grids for interleaved output
        uint32_t ncores = sub_core_grids->num_cores();
        TT_FATAL(ncores != 0, "number of cores cannot be 0");

        // Find the maximum number of cores that evenly divides num_output_pages
        for (uint32_t core_id = ncores; core_id >= 1; core_id--) {
            if (num_output_pages % core_id == 0) {
                ncores = core_id;
                break;
            }
            ncores--;
        }
        TT_FATAL(
            (num_output_pages % ncores == 0),
            "{} num of pages are not split uniformly across {} num of cores",
            num_output_pages,
            ncores);

        cores_list = corerange_to_cores(sub_core_grids.value(), ncores, rm_orientation);
        all_cores =
            num_cores_to_corerangeset_in_subcoregrids(cores_list[0], ncores, sub_core_grids.value(), rm_orientation);
        if (ncores == 1) {
            all_cores = ttnn::CoreRangeSet(ttnn::CoreRange(cores_list[0]));
        }
        num_cores = ncores;
        num_tiles_per_core_group_1 = num_output_pages / ncores;
        num_tiles_per_core_group_2 = 0;
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
    } else {
        // Use full compute grid
        const CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        num_cores_x = compute_with_storage_grid_size.x;
        num_cores_y = compute_with_storage_grid_size.y;
        auto
            [num_cores_result,
             all_cores_result,
             core_group_1_result,
             core_group_2_result,
             num_tiles_per_core_group_1_result,
             num_tiles_per_core_group_2_result] =
                split_work_to_cores(compute_with_storage_grid_size, num_output_pages, rm_orientation);
        num_cores = num_cores_result;
        all_cores = all_cores_result;
        core_group_1 = core_group_1_result;
        core_group_2 = core_group_2_result;
        num_tiles_per_core_group_1 = num_tiles_per_core_group_1_result;
        num_tiles_per_core_group_2 = num_tiles_per_core_group_2_result;
    }

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
    std::vector<uint32_t> num_pages_in_row_per_tensor(num_input_tensors, 1);
    std::vector<uint32_t> size_of_valid_data_in_last_page_per_tensor(num_input_tensors);

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
            const auto& in = input_tensors[i];
            auto* buffer = in.buffer();
            src_addr[i] = buffer->address();
            page_size_per_tensor[i] = buffer->page_size();
            size_of_valid_data_in_last_page_per_tensor[i] = buffer->page_size();
            if (in.is_sharded()) {
                const uint32_t shard_width = in.shard_spec().has_value() ? in.shard_spec().value().shape[1]
                                                                         : in.nd_shard_spec().value().shard_shape[-1];
                const uint32_t logical_width = in.logical_shape()[-1];
                num_pages_in_row_per_tensor[i] = tt::div_up(logical_width, shard_width);
                const uint32_t unpadded_row_bytes = logical_width * in.element_size();
                size_of_valid_data_in_last_page_per_tensor[i] =
                    unpadded_row_bytes - (num_pages_in_row_per_tensor[i] - 1) * buffer->page_size();
            }
            if (dim == num_dims - 1) {
                num_pages_per_block[i] = num_accum_pages * num_pages_in_row_per_tensor[i];
            } else {
                uint32_t dim_pages = in.padded_shape()[dim];
                num_pages_per_block[i] = num_accum_pages * dim_pages * num_pages_in_row_per_tensor[i];
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

    // Reader compile-time args (for RM: include per-tensor page params for sharded/ND-sharded inputs)
    std::vector<uint32_t> reader_compile_time_args = {src0_cb_index, num_input_tensors};
    reader_compile_time_args.insert(
        reader_compile_time_args.end(), page_size_per_tensor.cbegin(), page_size_per_tensor.cend());
    if (rm_layout) {
        reader_compile_time_args.insert(
            reader_compile_time_args.end(), num_pages_in_row_per_tensor.cbegin(), num_pages_in_row_per_tensor.cend());
        reader_compile_time_args.insert(
            reader_compile_time_args.end(),
            size_of_valid_data_in_last_page_per_tensor.cbegin(),
            size_of_valid_data_in_last_page_per_tensor.cend());
    }
    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        TensorAccessorArgs(*input_tensors[i].buffer()).append_to(reader_compile_time_args);
    }

    std::map<std::string, std::string> concat_defines;

    if (rm_layout && dim == num_dims - 1) {
        concat_defines["WIDTH_CONCAT"] = "1";
    }

    uint32_t num_pages_in_row_out = 1;
    uint32_t size_of_valid_data_in_last_page_out = dst_buffer->page_size();
    const bool output_sharded_rm = rm_layout && output.is_sharded();
    if (output_sharded_rm) {
        const uint32_t shard_width = output.shard_spec().has_value() ? output.shard_spec().value().shape[1]
                                                                     : output.nd_shard_spec().value().shard_shape[-1];
        const uint32_t logical_width = output.logical_shape()[-1];
        num_pages_in_row_out = tt::div_up(logical_width, shard_width);
        const uint32_t unpadded_row_bytes = logical_width * output.element_size();
        size_of_valid_data_in_last_page_out = unpadded_row_bytes - (num_pages_in_row_out - 1) * dst_buffer->page_size();
        TT_FATAL(
            dst_buffer->page_size() == dst_buffer->aligned_page_size(),
            "Output row-major shard width {} gives page size {} bytes, which must be aligned to {} bytes",
            shard_width,
            dst_buffer->page_size(),
            hal::get_l1_alignment());
    }

    std::vector<uint32_t> writer_compile_time_args;
    if (rm_layout) {
        if (output_sharded_rm) {
            writer_compile_time_args = {
                (std::uint32_t)src0_cb_index,
                dst_buffer->page_size(),
                num_pages_in_row_out,
                size_of_valid_data_in_last_page_out};
        } else {
            writer_compile_time_args = {(std::uint32_t)src0_cb_index, dst_buffer->page_size()};
        }
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

    const bool use_concat_writer = rm_layout && output.is_sharded();
    writer_kernel_id = CreateKernel(
        program,
        use_concat_writer
            ? "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/writer_concat_stick_layout.cpp"
            : (rm_layout ? "ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp"
                         : "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                           "writer_unary_interleaved_start_id.cpp"),
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    if (sub_core_grids.has_value() && !output.is_sharded()) {
        // Use the cores list we already computed from sub_core_grids
        cores = cores_list;
    } else {
        // Use grid_to_cores for full compute grid
        cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, rm_orientation);
    }
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

        const uint32_t writer_start_page_id =
            rm_layout && output.is_sharded() ? num_pages_written * num_pages_in_row_out : num_pages_written;
        std::vector<uint32_t> writer_kernel_args;
        if (rm_layout) {
            writer_kernel_args = {dst_buffer->address(), single_page_size, num_pages_per_core, writer_start_page_id};
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
