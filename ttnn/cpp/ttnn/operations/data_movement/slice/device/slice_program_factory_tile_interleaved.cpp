// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile_interleaved.hpp"

#include <optional>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

// Slice Tile Interleaved Program Factory implementation
SliceTileInterleavedProgramFactory::cached_program_t SliceTileInterleavedProgramFactory::create(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    tt::tt_metal::IDevice* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t single_tile_size = tt::tile_size(output_cb_data_format);

    uint32_t num_tiles = output.physical_volume() / TILE_HW;
    uint32_t num_tiles_per_row = output.padded_shape()[-1] / TILE_WIDTH;
    uint32_t num_tiles_per_col = output.padded_shape()[-2] / TILE_HEIGHT;
    auto rank = output.logical_shape().rank();

    // Check if the circular buffer (CB) can accommodate at least two rows or columns of tiles for double buffering.
    bool enough_space_width = ttnn::operations::data_movement::is_enough_space(
        output, single_tile_size, single_tile_size, 2 * num_tiles_per_col);
    bool enough_space_height = ttnn::operations::data_movement::is_enough_space(
        output, single_tile_size, single_tile_size, 2 * num_tiles_per_row);

    bool row_wise = false;  // one block consists of one tile column
    if (enough_space_height && num_tiles_per_row >= num_tiles_per_col) {
        row_wise = true;  // one block consists of one tile row
    } else if (enough_space_width) {
        row_wise = false;
    } else {
        TT_FATAL(false, "neither enough space along both dimensions");
    }

    auto num_tiles_per_block = row_wise ? num_tiles_per_row : num_tiles_per_col;
    auto num_blocks = num_tiles / num_tiles_per_block;

    auto [num_cores, all_cores, core_group, core_group_cliff, num_blocks_per_core, num_blocks_per_core_cliff] =
        args.sub_core_grids.has_value() ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_blocks)
                                        : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

    // make circular buffer
    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(
            2 * num_tiles_per_block * single_tile_size, {{tt::CBIndex::c_0, cb_data_format}})
            .set_page_size(tt::CBIndex::c_0, single_tile_size);

    if (!core_group.empty()) {
        tt::tt_metal::CreateCircularBuffer(program, core_group, cb_config);
    }
    if (!core_group_cliff.empty()) {
        tt::tt_metal::CreateCircularBuffer(program, core_group_cliff, cb_config);
    }

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // reader
    uint32_t src_tiles_per_row = input.padded_shape()[-1] / TILE_WIDTH;
    uint32_t src_tiles_per_col = input.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t src_tile_stride = row_wise ? 1 : src_tiles_per_row;
    uint32_t src_block_stride = row_wise ? src_tiles_per_row : 1;

    std::vector<uint32_t> reader_compile_time_args = {
        0 /*tt::CBIndex::c_0*/, num_tiles_per_block, src_tile_stride, src_block_stride, rank - 1, single_tile_size};
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "slice_reader_unary_tile_row_col_interleaved.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // writer
    uint32_t out_tiles_per_row = output.padded_shape()[-1] / TILE_WIDTH;
    uint32_t out_tiles_per_col = output.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t out_tile_stride = row_wise ? 1 : out_tiles_per_row;
    uint32_t out_block_stride = row_wise ? out_tiles_per_row : 1;

    std::vector<uint32_t> writer_compile_time_args = {
        0 /*tt::CBIndex::c_0*/, num_tiles_per_block, out_tile_stride, out_block_stride, rank - 1, single_tile_size};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "slice_writer_unary_tile_row_col_interleaved.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    // RUNTIME ARGS
    std::vector<uint32_t> src_shape_blocks(rank - 1);
    std::vector<uint32_t> out_shape_blocks(rank - 1);
    std::vector<uint32_t> src_block_id_gap(rank - 1);
    std::vector<uint32_t> out_block_id_gap(rank - 1);
    std::vector<uint32_t> block_coord(rank - 1, 0);

    int index = rank - 2;
    if (row_wise) {
        out_shape_blocks[index] = out_tiles_per_col;
        src_shape_blocks[index] = src_tiles_per_col;
        src_block_id_gap[index] = src_tiles_per_row * (src_tiles_per_col - out_tiles_per_col);
        out_block_id_gap[index] = 0;
        index--;
    } else {
        out_shape_blocks[index] = out_tiles_per_row;
        src_shape_blocks[index] = src_tiles_per_row;
        src_block_id_gap[index] = src_tiles_per_row * src_tiles_per_col - out_tiles_per_row;
        out_block_id_gap[index] = out_tiles_per_row * out_tiles_per_col - out_tiles_per_row;
        index--;
    }
    uint32_t size_acc_src = src_tiles_per_row * src_tiles_per_col;
    for (int32_t i = rank - 3; i >= 0; i--) {
        int size_unpadded_dim = input.padded_shape()[i] - output.padded_shape()[i];
        out_shape_blocks[index] = output.padded_shape()[i];
        src_shape_blocks[index] = input.padded_shape()[i];
        src_block_id_gap[index] = size_unpadded_dim * size_acc_src;
        out_block_id_gap[index] = 0;
        size_acc_src *= input.padded_shape()[i];
        index--;
    }

    auto calc_block_tile_id = [](const std::vector<uint32_t>& coord,
                                 const std::vector<uint32_t>& shape,
                                 const std::vector<uint32_t>& zero_index,
                                 const bool row_wise,
                                 const uint32_t sdim_size,
                                 const uint32_t sdim_zero_index) -> uint32_t {
        uint32_t index = coord.size() - 1;
        uint32_t tile_index = 0;
        uint32_t multiplier = 1;
        if (row_wise) {
            tile_index += sdim_zero_index * multiplier;
            multiplier *= sdim_size;
            tile_index += (coord[index] + zero_index[index]) * multiplier;
            multiplier *= shape[index];
        } else {
            tile_index += (coord[index] + zero_index[index]) * multiplier;
            multiplier *= shape[index];
            tile_index += sdim_zero_index * multiplier;
            multiplier *= sdim_size;
        }
        index--;
        for (int i = coord.size() - 2; i >= 0; i--) {
            tile_index += (coord[i] + zero_index[i]) * multiplier;
            multiplier *= shape[i];
            index--;
        }
        return tile_index;
    };

    std::vector<uint32_t> block_start(rank - 1);
    std::vector<uint32_t> zero_coord(rank - 1, 0);
    if (row_wise) {
        block_start[rank - 2] = args.slice_start[rank - 2] / TILE_HEIGHT;
    } else {
        block_start[rank - 2] = args.slice_start[rank - 1] / TILE_WIDTH;
    }
    for (int i = 0; i < rank - 2; i++) {
        block_start[i] = args.slice_start[i];
    }
    uint32_t src_block_dim_start =
        row_wise ? args.slice_start[rank - 1] / TILE_WIDTH : args.slice_start[rank - 2] / TILE_HEIGHT;
    uint32_t src_block_dim_size = row_wise ? src_tiles_per_row : src_tiles_per_col;
    uint32_t out_block_dim_size = row_wise ? out_tiles_per_row : out_tiles_per_col;

    const auto cores = corerange_to_cores(all_cores);
    for (uint32_t i = 0; i < num_cores; i++) {
        const auto& core = cores[i];
        uint32_t num_blocks_arg = num_blocks_per_core;
        if (i >= core_group.num_cores()) {
            num_blocks_arg = num_blocks_per_core_cliff;
        }

        // Validate block_coord doesn't exceed src_shape_blocks bounds before calculating source tile_id
        // This prevents NOC address overflow from invalid tile_id calculations
        for (uint32_t dim = 0; dim < block_coord.size(); ++dim) {
            if (block_coord[dim] >= src_shape_blocks[dim]) {
                TT_FATAL(
                    false,
                    "block_coord[{}] ({}) >= src_shape_blocks[{}] ({}). "
                    "This indicates a bug in block_coord calculation or wrapping logic.",
                    dim,
                    block_coord[dim],
                    dim,
                    src_shape_blocks[dim]);
            }
        }
        uint32_t src_block_tile_id = calc_block_tile_id(
            block_coord, src_shape_blocks, block_start, row_wise, src_block_dim_size, src_block_dim_start);
        uint32_t out_block_tile_id =
            calc_block_tile_id(block_coord, out_shape_blocks, zero_coord, row_wise, out_block_dim_size, 0);

        std::vector<uint32_t> reader_rt_args = {src_buffer->address(), src_block_tile_id, num_blocks_arg};
        reader_rt_args.insert(reader_rt_args.end(), out_shape_blocks.begin(), out_shape_blocks.end());
        reader_rt_args.insert(reader_rt_args.end(), src_block_id_gap.begin(), src_block_id_gap.end());
        reader_rt_args.insert(reader_rt_args.end(), block_coord.begin(), block_coord.end());

        std::vector<uint32_t> writer_rt_args = {dst_buffer->address(), out_block_tile_id, num_blocks_arg};
        writer_rt_args.insert(writer_rt_args.end(), out_shape_blocks.begin(), out_shape_blocks.end());
        writer_rt_args.insert(writer_rt_args.end(), out_block_id_gap.begin(), out_block_id_gap.end());
        writer_rt_args.insert(writer_rt_args.end(), block_coord.begin(), block_coord.end());

        SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);

        block_coord[rank - 2] += num_blocks_arg;
        for (int j = rank - 2; j >= 1; j--) {
            if (block_coord[j] >= out_shape_blocks[j]) {
                const uint32_t carry = block_coord[j] / out_shape_blocks[j];
                block_coord[j] = block_coord[j] % out_shape_blocks[j];
                block_coord[j - 1] += carry;
            } else {
                break;
            }
        }
    }

    return {std::move(program), {unary_reader_kernel_id, unary_writer_kernel_id, cores, num_cores}};
}

void SliceTileInterleavedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program, const SliceParams& /*args*/, const SliceInputs& tensor_args, Tensor& output) {
    auto& program = cached_program.program;
    auto* src_buffer = tensor_args.input.buffer();
    auto* dst_buffer = output.buffer();
    const auto& cores = cached_program.shared_variables.cores;
    const auto num_cores = cached_program.shared_variables.ncores;

    for (uint32_t i = 0; i < num_cores; ++i) {
        {
            auto& runtime_args =
                GetRuntimeArgs(program, cached_program.shared_variables.unary_reader_kernel_id, cores[i]);
            runtime_args[0] = src_buffer->address();
        }
        {
            auto& runtime_args =
                GetRuntimeArgs(program, cached_program.shared_variables.unary_writer_kernel_id, cores[i]);
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::prim
