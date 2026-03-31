// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_untilize_program_factory.hpp"

#include <algorithm>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"

#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::prim {

ConcatUntilizeProgramFactory::cached_program_t ConcatUntilizeProgramFactory::create(
    const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;

    const auto& input_tensors = tensor_args.input_tensors;
    const uint32_t dim = operation_attributes.dim;
    const Tensor& output = tensor_return_value;

    Program program = CreateProgram();

    IDevice* device = output.device();
    const auto grid_size = device->compute_with_storage_grid_size();

    // Input tensor properties (all inputs must have same dtype/tile)
    const auto& first_input = input_tensors[0];
    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(first_input.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    // Output tensor properties (ROW_MAJOR, potentially different dtype for BFP8→BF16)
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);
    uint32_t output_element_size = output.element_size();

    // Tile dimensions
    uint32_t tile_width = first_input.tensor_spec().tile().get_tile_shape()[1];
    uint32_t tile_height = first_input.tensor_spec().tile().get_tile_shape()[0];

    // Output dimensions
    uint32_t output_width = output.padded_shape()[-1];
    uint32_t output_height = output.physical_volume() / output_width;
    uint32_t num_tiles_per_row = output_width / tile_width;
    uint32_t num_tile_rows = output_height / tile_height;

    // Distribute tile-rows across cores
    auto
        [num_compute_cores,
         compute_core_range,
         full_compute_core_range,
         cliff_compute_core_range,
         num_rows_per_full_core,
         num_rows_per_cliff_core] = ttnn::split_blocks_for_tilize(grid_size, num_tile_rows);

    // Input CB: only needs to hold enough tiles for one sub-block (what DEST can process at once)
    // The untilize compute kernel waits for sub_block_width tiles, processes them, then pops.
    // Double-buffer to overlap reader DMA with compute processing.
    bool fp32_dest_acc_en = first_input.dtype() == DataType::UINT32 || first_input.dtype() == DataType::FLOAT32;
    uint32_t dest_limit = fp32_dest_acc_en ? 4 : 8;  // DEST_AUTO_LIMIT depends on data format
    uint32_t sub_block_width = dest_limit;
    // Find largest divisor of num_tiles_per_row that fits in DEST
    for (uint32_t w = dest_limit; w >= 1; --w) {
        if (num_tiles_per_row % w == 0) {
            sub_block_width = w;
            break;
        }
    }
    uint32_t input_cb_num_tiles = sub_block_width * 2;  // double-buffered sub-blocks
    auto [src0_cb_index, cb_src0] = create_cb(
        tt::CBIndex::c_0,
        program,
        compute_core_range,
        input_single_tile_size,
        input_cb_num_tiles,
        input_cb_data_format);

    // Output CB: holds one full untilized tile-row (num_tiles_per_row pages)
    // pack_untilize_block writes sub-blocks into positions within this row
    uint32_t output_cb_num_tiles = num_tiles_per_row;
    auto [output_cb_index, cb_output] = create_cb(
        tt::CBIndex::c_16,
        program,
        compute_core_range,
        output_single_tile_size,
        output_cb_num_tiles,
        output_cb_data_format);

    // ========================================
    // Reader kernel: concat tile reader (reads tiles from multiple input tensors)
    // ========================================
    const uint32_t num_input_tensors = input_tensors.size();

    std::vector<uint32_t> page_size_per_tensor(num_input_tensors);
    std::vector<uint32_t> num_pages_per_block(num_input_tensors);

    // For dim=0 concat, compute pages per block for each tensor
    uint32_t num_accum_pages = 1;
    uint32_t scale_factor = 1;
    uint32_t num_dims = output.padded_shape().rank();

    if (dim == num_dims - 2) {
        scale_factor = tile_height;
    } else if (dim == num_dims - 1) {
        scale_factor = tile_width;
    }

    for (uint32_t i = dim + 1; i < num_dims; ++i) {
        num_accum_pages *= output.padded_shape()[i];
    }
    if (dim < num_dims - 2) {
        num_accum_pages /= TILE_HW;
    } else if (dim == num_dims - 2) {
        num_accum_pages /= tile_width;
    }

    uint32_t num_output_pages_per_block = 0;
    std::vector<uint32_t> src_addr(num_input_tensors);
    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        auto* buffer = input_tensors[i].buffer();
        src_addr[i] = buffer->address();
        page_size_per_tensor[i] = buffer->page_size();
        uint32_t dim_pages = input_tensors[i].padded_shape()[dim] / scale_factor;
        num_pages_per_block[i] = num_accum_pages * dim_pages;
        num_output_pages_per_block += num_accum_pages * dim_pages;
    }

    // Reader compile-time args
    std::vector<uint32_t> reader_compile_time_args = {src0_cb_index, num_input_tensors};
    reader_compile_time_args.insert(
        reader_compile_time_args.end(), page_size_per_tensor.cbegin(), page_size_per_tensor.cend());
    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        TensorAccessorArgs(*input_tensors[i].buffer()).append_to(reader_compile_time_args);
    }

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
        "reader_concat_interleaved_start_id.cpp",
        compute_core_range,
        ReaderDataMovementConfig(reader_compile_time_args));

    // ========================================
    // Compute kernel: untilize
    // ========================================
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    std::map<std::string, std::string> compute_kernel_defines;
    if (first_input.dtype() == DataType::INT32 || first_input.dtype() == DataType::UINT32 ||
        first_input.dtype() == DataType::FLOAT32) {
        compute_kernel_defines["DST_ACCUM_MODE"] = "1";
    }

    std::vector<uint32_t> compute_compile_time_args = {num_tiles_per_row, src0_cb_index, output_cb_index};

    KernelHandle compute_kernel_id = 0;
    if (!full_compute_core_range.ranges().empty()) {
        compute_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/"
            "untilize_variable_num_blocks.cpp",
            full_compute_core_range,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .compile_args = compute_compile_time_args,
                .defines = compute_kernel_defines});
    }

    KernelHandle compute_cliff_kernel_id = 0;
    if (!cliff_compute_core_range.ranges().empty()) {
        compute_cliff_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/"
            "untilize_variable_num_blocks.cpp",
            cliff_compute_core_range,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .compile_args = compute_compile_time_args,
                .defines = compute_kernel_defines});
    }

    // ========================================
    // Writer kernel: row-major stick writer
    // ========================================
    uint32_t output_stick_size = output_width * output_element_size;
    uint32_t num_cols_per_input_block = num_tiles_per_row * tile_width;
    uint32_t num_cols_per_output_block = output_width;  // interleaved: one full row per page

    std::vector<uint32_t> writer_compile_time_args = {
        output_cb_index,
        output_stick_size,
        tile_height,
        num_tiles_per_row,
        1,  // output_num_blocks_across_width (interleaved = 1)
        output_element_size,
        num_cols_per_input_block,
        num_cols_per_output_block,
    };
    TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
        "writer_unary_stick_layout_split_rows_multi_core.cpp",
        compute_core_range,
        WriterDataMovementConfig(writer_compile_time_args));

    // ========================================
    // Runtime args per core
    // ========================================
    std::vector<CoreCoord> cores = corerange_to_cores(compute_core_range, std::nullopt, true);
    uint32_t g1_num_cores = full_compute_core_range.num_cores();
    uint32_t tile_start_index = 0;

    for (uint32_t i = 0; i < cores.size(); ++i) {
        CoreCoord core = cores[i];
        uint32_t num_tile_rows_this_core = (i < g1_num_cores) ? num_rows_per_full_core : num_rows_per_cliff_core;
        uint32_t num_tiles_this_core = num_tiles_per_row * num_tile_rows_this_core;

        // Determine starting tensor and position within concat blocks
        uint32_t block_id = tile_start_index / num_output_pages_per_block;
        uint32_t id_within_block = tile_start_index % num_output_pages_per_block;
        uint32_t curr_tensor = 0;
        uint32_t curr_tensor_id = 0;
        std::vector<uint32_t> page_id_per_tensor(num_input_tensors);

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

        // Reader runtime args
        std::vector<uint32_t> reader_kernel_args = {num_tiles_this_core, curr_tensor, curr_tensor_id};
        reader_kernel_args.insert(reader_kernel_args.end(), src_addr.cbegin(), src_addr.cend());
        reader_kernel_args.insert(reader_kernel_args.end(), num_pages_per_block.cbegin(), num_pages_per_block.cend());
        reader_kernel_args.insert(reader_kernel_args.end(), page_id_per_tensor.begin(), page_id_per_tensor.end());

        // Writer runtime args
        uint32_t height_wise_input_block_start_index = tile_start_index / num_tiles_per_row;
        std::vector<uint32_t> writer_run_time_args = {
            output.buffer()->address(),
            num_tile_rows_this_core,
            height_wise_input_block_start_index,
            num_cols_per_input_block,  // num_unpadded_cols_per_input_block
            0,                         // width_wise_output_block_start_index
            0,                         // num_cols_already_processed_in_first_output_block
        };

        // Compute runtime args
        std::vector<uint32_t> compute_run_time_args = {num_tile_rows_this_core};

        SetRuntimeArgs(program, reader_kernel_id, core, reader_kernel_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_run_time_args);
        if (i < g1_num_cores) {
            SetRuntimeArgs(program, compute_kernel_id, core, compute_run_time_args);
        } else {
            SetRuntimeArgs(program, compute_cliff_kernel_id, core, compute_run_time_args);
        }

        tile_start_index += num_tiles_this_core;
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, compute_kernel_id, cores}};
}

void ConcatUntilizeProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ConcatParams& /*operation_attributes*/,
    const ConcatInputs& tensor_args,
    Tensor& tensor_return_value) {
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
