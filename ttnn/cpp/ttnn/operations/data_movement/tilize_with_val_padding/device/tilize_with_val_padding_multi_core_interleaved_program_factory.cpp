// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_multi_core_interleaved_program_factory.hpp"
#include "tilize_with_val_padding_single_core_program_factory.hpp"
#include "tilize_with_val_padding_multi_core_block_interleaved_program_factory.hpp"

#include <math.h>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding_common.hpp"
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

tt::tt_metal::operation::ProgramWithCallbacks tilize_with_val_padding_multi_core_interleaved(
    const Tensor& a, Tensor& output, const ttnn::PadValue pad_value) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    IDevice* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();

    uint32_t num_blocks = output.physical_volume() / output.padded_shape()[-1] / TILE_HEIGHT;
    uint32_t num_tiles_per_row = output.padded_shape()[-1] / TILE_WIDTH;

    uint32_t num_tiles_per_col = output.padded_shape()[-2] / TILE_HEIGHT;
    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(grid_size, num_blocks);

    constexpr uint32_t threshold_row_block = 32;
    if (num_tiles_per_row > threshold_row_block) {
        if (num_tiles_per_col > threshold_row_block || num_tiles_per_row > num_tiles_per_col) {
            uint32_t num_blocks_block = (a.padded_shape()[-1] * a.padded_shape()[-2]) / (TILE_HEIGHT * TILE_WIDTH);

            auto
                [ncores_block,
                 all_cores_block,
                 core_range_block,
                 cliff_row_core_range,
                 cliff_col_core_range,
                 cliff_col_row_core_range,
                 nblocks_per_core_block,
                 single_block_size,
                 single_block_size_cliff_row,
                 single_block_size_cliff_col,
                 has_cliff_row,
                 has_cliff_col,
                 full_cores_per_row,
                 full_cores_per_col] =
                    ttnn::split_blocks_for_tilize_wh(grid_size, num_blocks_block, num_tiles_per_row, num_tiles_per_col);
            if (ncores < ncores_block) {
                return tilize_with_val_padding_multi_core_block_interleaved(a, output, pad_value);
            }
        }
    }

    bool has_cliff = core_range_cliff.size() > 0;

    uint32_t unpadded_row_size_bytes = a.padded_shape()[-1] * a.element_size();     // Assuming bfloat16 dataformat
    uint32_t padded_row_size_bytes = output.padded_shape()[-1] * a.element_size();  // Assuming bfloat16 dataformat

    auto [src0_cb_index, cb_src0] = create_cb(
        tt::CBIndex::c_0, program, all_cores, input_single_tile_size, num_tiles_per_row, input_cb_data_format);

    auto [output_cb_index, cb_output] = create_cb(
        tt::CBIndex::c_16, program, all_cores, output_single_tile_size, num_tiles_per_row, output_cb_data_format);

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /** reader
     */
    uint32_t src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    uint32_t stick_size = unpadded_row_size_bytes;
    uint32_t stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)std::log2(stick_size) : 0;

    uint32_t packed_pad_value = get_packed_value(a, pad_value);
    // log2(TILE_WIDTH * data_format_size_in_bytes)
    uint32_t shift_bits = (a.dtype() == DataType::BFLOAT16) ? 6 : 7;

    KernelHandle unary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/"
        "reader_unary_pad_dims_split_rows_multicore.cpp",
        all_cores,
        ReaderDataMovementConfig({src0_is_dram, stick_size_is_power_of_two, log2_stick_size, shift_bits}));

    /** writer
     */
    uint32_t out_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        WriterDataMovementConfig({output_cb_index, out_is_dram}));

    /** compute
     */
    if (core_range.size() > 0) {
        auto tilize_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp",
            core_range,
            ComputeConfig{.compile_args = {nblocks_per_core, num_tiles_per_row}});
    }
    if (has_cliff) {
        auto tilize_cliff_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp",
            core_range_cliff,
            ComputeConfig{.compile_args = {nblocks_per_core_cliff, num_tiles_per_row}});
    }

    /* RUNTIME ARGS */
    // 1D distribution of blocks across cores
    uint32_t tile_height = output.tensor_spec().tile().get_height();
    auto core_assignments = ttnn::distribute_work(
        output.logical_shape(),
        output.padded_shape(),
        ncores,
        nblocks_per_core,
        has_cliff,
        nblocks_per_core_cliff,
        tile_height);

    uint32_t tile_start_id = 0;
    uint32_t row_start_id = 0;
    uint32_t ncores_x = grid_size.x;

    const auto& cores = grid_to_cores(ncores, grid_size.x, grid_size.y, true);
    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];
        const std::vector<BlockRep>& assignment = core_assignments.at(i);

        // reader runtime args
        std::vector<uint32_t> reader_rt_args = {
            src0_buffer->address(),
            unpadded_row_size_bytes,
            padded_row_size_bytes,
            packed_pad_value,
            row_start_id,
            static_cast<unsigned int>(assignment.size()),
        };

        uint32_t nblocks_per_core = 0;
        BlockRep ref_el = assignment[0];
        uint32_t count_repeated = 0;  // will be incremented in first iteration of the loop
        for (const auto& el : assignment) {
            nblocks_per_core += el.block_count();
            row_start_id += el.data_row_count();
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

    auto override_runtime_args_callback =
        [reader_kernel_id = unary_reader_kernel_id, writer_kernel_id = unary_writer_kernel_id, cores = cores](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto src_buffer = input_tensors.at(0).buffer();
            auto dst_buffer = output_tensors.at(0).buffer();

            auto& reader_runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
            auto& writer_runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
            for (const auto& core : cores) {
                {
                    auto& runtime_args = reader_runtime_args_by_core[core.x][core.y];
                    runtime_args[0] = src_buffer->address();
                }
                {
                    auto& runtime_args = writer_runtime_args_by_core[core.x][core.y];
                    runtime_args[0] = dst_buffer->address();
                }
            }
        };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::data_movement::detail
