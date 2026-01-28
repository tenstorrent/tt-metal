// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_multi_core_width_sharded_program_factory.hpp"
#include "ttnn/operations/cb_utils.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

TilizeMultiCoreWidthShardedProgramFactory::cached_program_t TilizeMultiCoreWidthShardedProgramFactory::create(
    const ttnn::prim::TilizeParams& /*operation_attributes*/,
    const ttnn::prim::TilizeInputs& tensor_args,
    const Tensor& output_tensor) {
    tt::tt_metal::Program program{};

    auto input = tensor_args.input_tensor;
    const auto& output = output_tensor;

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    bool fp32_llk_acc = input.dtype() == DataType::FLOAT32;

    auto shard_spec = input.shard_spec().value();
    // uint32_t num_tiles_per_shard = shard_spec.shape[0] * shard_spec.shape[1] / TILE_HW;
    uint32_t num_tiles_per_row = input.logical_shape()[1] / TILE_WIDTH;

    auto* device = input.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange default_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    CoreRangeSet total_cores = CoreRangeSet(default_cores);
    uint32_t num_cores_total = total_cores.num_cores();

    // Calculate row width in bytes and datum size
    uint32_t row_width = input.logical_shape()[1] * input.element_size();
    // const uint32_t datum_size = input.element_size();
    uint32_t responsibility = ((num_tiles_per_row - 1) / num_cores_total) + 1;

    uint32_t src0_cb_index = 0;
    uint32_t src1_cb_index = 1;
    uint32_t page_size = single_tile_size;
    uint32_t cb_size = page_size * responsibility;

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, page_size);
    CBHandle cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(cb_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, page_size);
    CBHandle cb_src1 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src1_config);

    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_cb_index, row_width};
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)src1_cb_index, single_tile_size};

    tt::tt_metal::TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_width_sharded.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_width_sharded.cpp",
        total_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_args = {uint32_t(responsibility), uint32_t(1)};

    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp",
        total_cores,
        tt::tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_llk_acc,
            .compile_args = compute_args,
        });

    // Calculate runtime arguments for reader kernel
    uint32_t src_addr = input.buffer()->address();
    uint32_t height_per_core = input.logical_shape()[0] / TILE_HEIGHT;  // Number of tile-height rows per core
    uint32_t start_chunk_id = 0;                                   // Will need to be updated per core if needed

    std::vector<uint32_t> reader_runtime_args = {src_addr, height_per_core, responsibility, start_chunk_id};

    tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, total_cores, reader_runtime_args);

    uint32_t dst_addr = output.buffer()->address();
    uint32_t core_idx = 0;

    // Set per-core writer args based on tile responsibility
    for (auto core_range : total_cores.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord core = {x, y};

                // Calculate starting tile ID and number of tiles for this core
                uint32_t start_tile_id = core_idx * responsibility;
                uint32_t tiles_for_this_core = responsibility;

                // Handle the case where last core might have fewer tiles
                if (start_tile_id + tiles_for_this_core > num_tiles_per_row) {
                    tiles_for_this_core = num_tiles_per_row - start_tile_id;
                }

                std::vector<uint32_t> writer_runtime_args = {dst_addr, start_tile_id, tiles_for_this_core};

                tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);

                core_idx++;
            }
        }
    }
    return TilizeMultiCoreWidthShardedProgramFactory::cached_program_t{
        std::move(program), {unary_reader_kernel_id, unary_writer_kernel_id, cb_src0, cb_src1}};
}

void TilizeMultiCoreWidthShardedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ttnn::prim::TilizeParams& /*operation_attributes*/,
    const ttnn::prim::TilizeInputs& tensor_args,
    const Tensor& output_tensor) {
    auto& input = tensor_args.input_tensor;
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output_tensor.buffer();

    // Update circular buffer addresses
    UpdateDynamicCircularBufferAddress(
        cached_program.program, cached_program.shared_variables.input_cb_handle, *src_buffer);
    UpdateDynamicCircularBufferAddress(
        cached_program.program, cached_program.shared_variables.output_cb_handle, *dst_buffer);

    // Update reader kernel runtime arguments
    auto shard_spec = input.shard_spec().value();
    uint32_t num_tiles_per_row = shard_spec.shape[1] / TILE_WIDTH;
    uint32_t src_addr = src_buffer->address();
    uint32_t height_per_core = shard_spec.shape[0] / TILE_HEIGHT;
    uint32_t start_chunk_id = 0;

    // Update writer kernel runtime arguments
    uint32_t dst_addr = dst_buffer->address();
    auto* device = input.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1});
    CoreRangeSet total_cores = CoreRangeSet(default_cores);
    uint32_t num_cores_total = total_cores.num_cores();
    uint32_t responsibility = ((num_tiles_per_row - 1) / num_cores_total) + 1;
    uint32_t core_idx = 0;

    std::vector<uint32_t> reader_runtime_args = {src_addr, height_per_core, responsibility, start_chunk_id};

    for (auto core_range : total_cores.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord core = {x, y};

                uint32_t start_tile_id = core_idx * responsibility;
                uint32_t tiles_for_this_core = responsibility;

                // Handle the case where last core might have fewer tiles
                if (start_tile_id + tiles_for_this_core > num_tiles_per_row) {
                    tiles_for_this_core = num_tiles_per_row - start_tile_id;
                }

                std::vector<uint32_t> writer_runtime_args = {dst_addr, start_tile_id, tiles_for_this_core};

                tt::tt_metal::SetRuntimeArgs(
                    cached_program.program,
                    cached_program.shared_variables.unary_writer_kernel_id,
                    core,
                    writer_runtime_args);

                core_idx++;
            }
        }
    }
}
}  // namespace ttnn::prim
