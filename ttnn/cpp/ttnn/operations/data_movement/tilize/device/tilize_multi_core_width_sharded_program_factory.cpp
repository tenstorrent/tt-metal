// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
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
    uint32_t num_tiles_per_row = input.logical_shape()[1] / TILE_WIDTH;

    auto* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange default_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    CoreRangeSet total_cores = CoreRangeSet(default_cores);
    uint32_t num_cores_total = total_cores.num_cores();

    // Calculate row width in bytes and core responsibility
    uint32_t row_width = shard_spec.shape[1] * input.element_size();
    uint32_t datum_size = input.element_size();
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

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src0_cb_index, row_width, num_cores_total, datum_size};
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)src1_cb_index, single_tile_size};

    tt::tt_metal::TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/reader_unary_width_sharded.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/writer_unary_width_sharded.cpp",
        total_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_args = {uint32_t(src0_cb_index), uint32_t(src1_cb_index)};

    tt::tt_metal::KernelHandle unary_compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/tilize_compute_width_sharded.cpp",
        total_cores,
        tt::tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_llk_acc,
            .compile_args = compute_args,
        });

    // Calculate runtime arguments for reader kernel
    uint32_t src_addr = input.mesh_buffer()->address();
    uint32_t height_per_core = input.logical_shape()[0] / TILE_HEIGHT;

    uint32_t dst_addr = output.mesh_buffer()->address();
    uint32_t core_idx = 0;

    // Set per-core writer args based on tile responsibility
    for (auto core : corerange_to_cores(total_cores, std::nullopt)) {
        // Calculate starting tile ID and number of tiles for this core
        uint32_t start_tile_id = core_idx * responsibility;
        uint32_t tiles_for_this_core = responsibility;

        // Handle the case where last core might have fewer tiles
        if (start_tile_id + tiles_for_this_core > num_tiles_per_row) {
            tiles_for_this_core = num_tiles_per_row - start_tile_id;
        }

        std::vector<uint32_t> reader_runtime_args = {src_addr, height_per_core, tiles_for_this_core, core_idx};

        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);

        std::vector<uint32_t> writer_runtime_args = {dst_addr, start_tile_id, tiles_for_this_core};

        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);

        std::vector<uint32_t> compute_runtime_args = {tiles_for_this_core};

        tt::tt_metal::SetRuntimeArgs(program, unary_compute_kernel_id, core, compute_runtime_args);

        core_idx++;
    }
    return TilizeMultiCoreWidthShardedProgramFactory::cached_program_t{
        std::move(program), {unary_reader_kernel_id, unary_writer_kernel_id, cb_src0, cb_src1}};
}

void TilizeMultiCoreWidthShardedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ttnn::prim::TilizeParams& /*operation_attributes*/,
    const ttnn::prim::TilizeInputs& tensor_args,
    const Tensor& output_tensor) {
    const auto& input = tensor_args.input_tensor;
    tt::tt_metal::Buffer* src_buffer = input.buffer();
    tt::tt_metal::Buffer* dst_buffer = output_tensor.buffer();

    auto& shared_variables = cached_program.shared_variables;
    const auto& reader_kernel_id = shared_variables.unary_reader_kernel_id;
    const auto& writer_kernel_id = shared_variables.unary_writer_kernel_id;

    auto& program = cached_program.program;

    auto* device = input.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1});
    CoreRangeSet total_cores = CoreRangeSet(default_cores);

    for (auto core : corerange_to_cores(total_cores, std::nullopt)) {
        // auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
        // runtime_args[0] = src_buffer->address();
        // runtime_args[1] = dst_buffer->address();
        auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
        reader_runtime_args[0] = src_buffer->address();

        auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
        writer_runtime_args[0] = dst_buffer->address();
    }
}
}  // namespace ttnn::prim
