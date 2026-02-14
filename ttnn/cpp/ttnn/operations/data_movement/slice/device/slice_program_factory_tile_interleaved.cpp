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

    auto [num_cores, all_cores, core_group, core_group_cliff, num_tiles_per_core, num_tiles_per_core_cliff] =
        args.sub_core_grids.has_value() ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_tiles)
                                        : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

    uint32_t rank = static_cast<uint32_t>(input.padded_shape().rank());

    // make circular buffer
    const uint32_t num_tiles_per_block = 2;
    const uint32_t num_blocks_per_cb = 4;
    uint32_t total_tiles_in_cb = num_blocks_per_cb * num_tiles_per_block;
    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(total_tiles_in_cb * single_tile_size, {{tt::CBIndex::c_0, cb_data_format}})
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
    uint32_t src_tile_stride = num_cores;
    std::vector<uint32_t> reader_compile_time_args = {0 /*tt::CBIndex::c_0*/, src_tile_stride, rank, single_tile_size};
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "slice_reader_unary_tile_interleaved.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // writer
    uint32_t out_tile_stride = num_cores;
    std::vector<uint32_t> writer_compile_time_args = {0 /*tt::CBIndex::c_0*/, out_tile_stride, rank, single_tile_size};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "slice_writer_unary_tile_interleaved.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    // RUNTIME ARGS
    uint32_t out_tiles_per_row = output.padded_shape()[-1] / TILE_WIDTH;
    uint32_t out_tiles_per_col = output.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t src_tiles_per_row = input.padded_shape()[-1] / TILE_WIDTH;
    uint32_t src_tiles_per_col = input.padded_shape()[-2] / TILE_HEIGHT;

    std::vector<uint32_t> src_shape_tiles(rank);
    std::vector<uint32_t> out_shape_tiles(rank);
    std::vector<uint32_t> src_coord_inc(rank, 0);
    std::vector<uint32_t> src_tile_id_acc(rank);
    std::vector<uint32_t> src_tile_start(rank);
    std::vector<uint32_t> out_tile_start(rank, 0);

    uint32_t size_acc_src = 1;
    for (int32_t i = rank - 1; i >= 0; i--) {
        src_tile_id_acc[i] = size_acc_src;
        if (i == static_cast<int32_t>(rank) - 1) {
            out_shape_tiles[i] = out_tiles_per_row;
            src_shape_tiles[i] = src_tiles_per_row;
            src_tile_start[i] = args.slice_start[i] / TILE_WIDTH;
            size_acc_src *= src_tiles_per_row;
        } else if (i == static_cast<int32_t>(rank) - 2) {
            out_shape_tiles[i] = out_tiles_per_col;
            src_shape_tiles[i] = src_tiles_per_col;
            src_tile_start[i] = args.slice_start[i] / TILE_HEIGHT;
            size_acc_src *= src_tiles_per_col;
        } else {
            out_shape_tiles[i] = output.padded_shape()[i];
            src_shape_tiles[i] = input.padded_shape()[i];
            src_tile_start[i] = args.slice_start[i];
            size_acc_src *= input.padded_shape()[i];
        }
    }
    src_coord_inc[rank - 1] = num_cores;
    for (int32_t i = static_cast<int32_t>(rank) - 1; i >= 1; i--) {
        if (src_coord_inc[i] >= out_shape_tiles[i]) {
            src_coord_inc[i - 1] += (src_coord_inc[i] / out_shape_tiles[i]);
            src_coord_inc[i] = src_coord_inc[i] % out_shape_tiles[i];
        }
    }

    auto coord_to_tile_id = [](const std::vector<uint32_t>& coord,
                               const std::vector<uint32_t>& shape,
                               const std::vector<uint32_t>& zero_index) -> uint32_t {
        uint32_t tile_id = 0;
        uint32_t multiplier = 1;
        for (int i = coord.size() - 1; i >= 0; i--) {
            tile_id += (coord[i] + zero_index[i]) * multiplier;
            multiplier *= shape[i];
        }
        return tile_id;
    };

    std::vector<uint32_t> tile_coord(rank, 0);
    const auto cores = corerange_to_cores(all_cores);
    for (uint32_t i = 0; i < num_cores; i++) {
        const auto& core = cores[i];
        uint32_t num_tiles_arg = num_tiles_per_core;
        if (i >= core_group.num_cores()) {
            num_tiles_arg = num_tiles_per_core_cliff;
        }

        uint32_t src_core_tile_id = coord_to_tile_id(tile_coord, src_shape_tiles, src_tile_start);
        uint32_t out_core_tile_id = coord_to_tile_id(tile_coord, out_shape_tiles, out_tile_start);

        std::vector<uint32_t> reader_rt_args = {src_buffer->address(), src_core_tile_id, num_tiles_arg};
        reader_rt_args.insert(reader_rt_args.end(), out_shape_tiles.begin(), out_shape_tiles.end());
        reader_rt_args.insert(reader_rt_args.end(), tile_coord.begin(), tile_coord.end());
        reader_rt_args.insert(reader_rt_args.end(), src_tile_id_acc.begin(), src_tile_id_acc.end());
        reader_rt_args.insert(reader_rt_args.end(), src_coord_inc.begin(), src_coord_inc.end());

        std::vector<uint32_t> writer_rt_args = {dst_buffer->address(), out_core_tile_id, num_tiles_arg};
        writer_rt_args.insert(writer_rt_args.end(), out_shape_tiles.begin(), out_shape_tiles.end());
        writer_rt_args.insert(writer_rt_args.end(), tile_coord.begin(), tile_coord.end());

        SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);

        tile_coord[rank - 1]++;
        for (int j = rank - 1; j >= 1; j--) {
            if (tile_coord[j] >= out_shape_tiles[j]) {
                tile_coord[j] = tile_coord[j] % out_shape_tiles[j];
                tile_coord[j - 1]++;
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
