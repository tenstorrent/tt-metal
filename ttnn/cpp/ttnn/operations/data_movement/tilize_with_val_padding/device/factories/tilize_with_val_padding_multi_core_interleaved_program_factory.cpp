// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_multi_core_interleaved_program_factory.hpp"

#include <cmath>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_factory_helper.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_multi_core_block_interleaved_program_factory.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

TilizeWithValPaddingMultiCoreInterleavedFactory::cached_program_t
TilizeWithValPaddingMultiCoreInterleavedFactory::create(
    const operation_attributes_t& operation_attributes, const Tensor& input_tensor, const Tensor& output_tensor) {
    std::cout << "Creating program for ND-sharded input\n";
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    const Tensor& a = input_tensor;
    const Tensor& output = output_tensor;
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32;

    IDevice* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid =
        operation_attributes.sub_core_grids.has_value() ? operation_attributes.sub_core_grids.value() : default_grid;
    uint32_t tile_width = output.tensor_spec().tile().get_width();
    uint32_t tile_height = output.tensor_spec().tile().get_height();
    uint32_t num_blocks = output.physical_volume() / output.padded_shape()[-1] / tile_height;
    uint32_t num_tiles_per_row = output.padded_shape()[-1] / tile_width;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(available_grid, num_blocks);

    bool has_cliff = !core_range_cliff.empty();

    uint32_t unpadded_row_size_bytes = a.logical_shape()[-1] * a.element_size();    // Assuming bfloat16 dataformat
    uint32_t padded_row_size_bytes = output.padded_shape()[-1] * a.element_size();  // Assuming bfloat16 dataformat

    create_cb(tt::CBIndex::c_0, program, all_cores, input_single_tile_size, num_tiles_per_row, input_cb_data_format);

    auto [output_cb_index, cb_output] = create_cb(
        tt::CBIndex::c_16, program, all_cores, output_single_tile_size, num_tiles_per_row, output_cb_data_format);

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /** reader
     */
    uint32_t packed_pad_value = detail::get_packed_value(a, operation_attributes.pad_value);
    // log2(tile_height * data_format_size_in_bytes)
    uint32_t shift_bits = static_cast<uint32_t>(std::log2(
        a.element_size() *
        tile_height));  // This gives log2 of bytes per tile row, so in the kernel we
                        // can shift right by this to get number of tiles.
                        // ex: bf16/uint16 -> log2(2 * 32) = 6, float32/int32/uint32 -> log2(4 * 32) = 7, etc.
    uint32_t elem_size = a.element_size();
    uint32_t num_sticks_in_row = 1;                 // tt::div_up(a.padded_shape()[-1], tile_width);
    uint32_t stick_size = unpadded_row_size_bytes;  // a.element_size() * tile_width;
    uint32_t size_of_valid_data_in_last_stick_in_row =
        unpadded_row_size_bytes;  // a.padded_shape()[-1] % tile_width * a.element_size();
    if (a.is_sharded()) {
        uint32_t shard_width =
            a.shard_spec().has_value() ? a.shard_spec().value().shape[1] : a.nd_shard_spec().value().shard_shape[-1];
        num_sticks_in_row = tt::div_up(a.logical_shape()[-1], shard_width);
        stick_size = a.element_size() * shard_width;
        std::cout << "AAAA unpadded_row_size_bytes: " << unpadded_row_size_bytes << "\n";
        std::cout << "AAAA stick_size: " << stick_size << "\n";
        std::cout << "AAAA num_sticks_in_row: " << num_sticks_in_row << "\n";
        size_of_valid_data_in_last_stick_in_row = unpadded_row_size_bytes - (num_sticks_in_row - 1) * stick_size;
    }
    std::cout << "num_sticks_in_row: " << num_sticks_in_row << "\n";
    std::cout << "stick_size: " << stick_size << "\n";
    std::cout << "size_of_valid_data_in_last_stick_in_row: " << size_of_valid_data_in_last_stick_in_row << "\n";
    std::cout << "unpadded_row_size_bytes: " << unpadded_row_size_bytes << "\n";
    std::cout << "elem_size: " << elem_size << "\n";
    std::cout << "shift_bits: " << shift_bits << "\n";

    std::vector<uint32_t> reader_compile_time_args = {
        shift_bits,
        unpadded_row_size_bytes,
        elem_size,
        num_sticks_in_row,
        stick_size,
        size_of_valid_data_in_last_stick_in_row};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    KernelHandle unary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/"
        "reader_unary_pad_dims_split_rows_multicore.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    /** writer
     */
    std::vector<uint32_t> writer_compile_time_args = {output_cb_index};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    /** compute
     */
    if (!core_range.empty()) {
        CreateKernel(
            program,
            "ttnn/cpp/ttnn/kernel/compute/tilize.cpp",
            core_range,
            ComputeConfig{.fp32_dest_acc_en = fp32_llk_acc, .compile_args = {nblocks_per_core, num_tiles_per_row}});
    }
    if (has_cliff) {
        CreateKernel(
            program,
            "ttnn/cpp/ttnn/kernel/compute/tilize.cpp",
            core_range_cliff,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_llk_acc, .compile_args = {nblocks_per_core_cliff, num_tiles_per_row}});
    }

    /* RUNTIME ARGS */
    // 1D distribution of blocks across cores
    auto core_assignments = ttnn::distribute_work(
        output.logical_shape(),
        output.padded_shape(),
        ncores,
        nblocks_per_core,
        has_cliff,
        nblocks_per_core_cliff,
        tile_height);

    uint32_t tile_start_id = 0;
    uint32_t start_page_id = 0;

    const auto cores = corerange_to_cores(available_grid);
    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];
        const std::vector<BlockRep>& assignment = core_assignments.at(i);

        // reader runtime args
        std::vector<uint32_t> reader_rt_args = {
            src0_buffer->address(),
            padded_row_size_bytes,
            packed_pad_value,
            start_page_id,
            static_cast<unsigned int>(assignment.size()),
        };

        uint32_t nblocks_per_core = 0;
        BlockRep ref_el = assignment[0];
        uint32_t count_repeated = 0;  // will be incremented in first iteration of the loop
        for (const auto& el : assignment) {
            nblocks_per_core += el.block_count();
            start_page_id += el.data_row_count() * num_sticks_in_row;
            if (compare_assignments(ref_el, el)) {
                count_repeated++;
            } else {
                // push back information for previous elements
                reader_rt_args.push_back(ref_el.n_data);
                reader_rt_args.push_back(ref_el.n_mixed);
                reader_rt_args.push_back(ref_el.n_pads);
                reader_rt_args.push_back(ref_el.times);
                reader_rt_args.push_back(count_repeated);
                // set up assignment for this element
                ref_el = el;
                count_repeated = 1;
            }
        }
        reader_rt_args.push_back(ref_el.n_data);
        reader_rt_args.push_back(ref_el.n_mixed);
        reader_rt_args.push_back(ref_el.n_pads);
        reader_rt_args.push_back(ref_el.times);
        reader_rt_args.push_back(count_repeated);

        uint32_t num_tiles_per_core = num_tiles_per_row * nblocks_per_core;

        // writer runtime args
        const std::array writer_rt_args = {dst_buffer->address(), num_tiles_per_core, tile_start_id};

        SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);

        tile_start_id += num_tiles_per_core;
    }

    shared_variables_t shared_variables{
        .reader_kernel_id = unary_reader_kernel_id,
        .writer_kernel_id = unary_writer_kernel_id,
        .cores = cores,
        .ncores = ncores};
    return cached_program_t(std::move(program), std::move(shared_variables));
}

void TilizeWithValPaddingMultiCoreInterleavedFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const Tensor& input_tensor,
    const Tensor& output_tensor) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;
    const auto& ncores = shared_variables.ncores;
    const auto& cores = shared_variables.cores;
    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();

    auto& reader_runtime_args_by_core = GetRuntimeArgs(program, shared_variables.reader_kernel_id);
    auto& writer_runtime_args_by_core = GetRuntimeArgs(program, shared_variables.writer_kernel_id);
    for (uint32_t i = 0; i < ncores; ++i) {
        const CoreCoord& core = cores[i];
        {
            auto& runtime_args = reader_runtime_args_by_core[core.x][core.y];
            runtime_args[0] = src_buffer->address();
        }
        {
            auto& runtime_args = writer_runtime_args_by_core[core.x][core.y];
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::prim
