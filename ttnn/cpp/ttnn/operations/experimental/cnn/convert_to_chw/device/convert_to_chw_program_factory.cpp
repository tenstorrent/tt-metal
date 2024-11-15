// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_chw_program_factory.hpp"

namespace ttnn::operations::experimental::cnn::detail {

using namespace tt::constants;

operation::ProgramWithCallbacks multi_core_convert_to_chw(
    const Tensor& a, Tensor& output, CoreCoord compute_with_storage_grid_size) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const auto input_shape = a.get_logical_shape();
    const auto input_core_grid = a.shard_spec()->grid;
    std::vector<CoreCoord> input_cores = grid_to_cores(
        input_core_grid.num_cores(), compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, true);

    const auto HW = input_shape[2];
    const auto C = input_shape[3];

    TT_ASSERT(C < TILE_HEIGHT, "C must be 32 or smaller");

    const uint32_t c_tiles = 1;  // assume C <= 32
    const uint32_t hw_tiles = HW / TILE_HEIGHT;
    const uint32_t total_tiles = hw_tiles * c_tiles;
    const uint32_t total_tiles_per_core = hw_tiles * c_tiles / input_cores.size();

    const auto create_circular_buffer = [&program, &input_core_grid](
                                            uint32_t index,
                                            uint32_t num_tiles,
                                            uint32_t tile_size,
                                            const tt::DataFormat& format,
                                            Buffer* buffer = nullptr) -> tt::tt_metal::CBHandle {
        auto config = CircularBufferConfig(num_tiles * tile_size, {{index, format}}).set_page_size(index, tile_size);
        if (buffer != nullptr) {
            config = config.set_globally_allocated_address(*buffer);
        }
        return tt::tt_metal::CreateCircularBuffer(program, input_core_grid, config);
    };

    const tt::DataFormat input_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    const uint32_t input_tile_size = tt::tt_metal::detail::TileSize(input_format);

    const tt::DataFormat intermediary_format = tt::DataFormat::Float16_b;
    const uint32_t intermediary_tile_size = tt::tt_metal::detail::TileSize(intermediary_format);

    const uint32_t cb_in_id = tt::CB::c_in0;
    const auto cb_in =
        create_circular_buffer(cb_in_id, total_tiles_per_core, input_tile_size, input_format, a.buffer());

    const uint32_t cb_out_id = tt::CB::c_out0;
    const auto cb_out =
        create_circular_buffer(cb_out_id, total_tiles_per_core, input_tile_size, input_format, output.buffer());

    const uint32_t cb_in_transpose_id = tt::CB::c_intermed0;
    const auto cb_in_transpose =
        create_circular_buffer(cb_in_transpose_id, 1, intermediary_tile_size, intermediary_format);

    std::vector<uint32_t> reader_compile_time_args = {cb_in_id};
    std::vector<uint32_t> writer_compile_time_args = {cb_in_transpose_id, cb_out_id, C};
    std::vector<uint32_t> compute_compile_time_args = {cb_in_id, cb_in_transpose_id, cb_out_id};

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw/device/kernels/reader_convert_to_chw.cpp",
        input_core_grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw/device/kernels/writer_convert_to_chw.cpp",
        input_core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw/device/kernels/convert_to_chw.cpp",
        input_core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args});

    auto set_runtime_args =
        [cb_in, cb_out, input_cores, total_tiles_per_core, reader_kernel_id, writer_kernel_id, compute_kernel_id](
            Program& program, const Tensor& a, const Tensor& output) {
            tt::tt_metal::Buffer* a_buffer = a.buffer();
            tt::tt_metal::Buffer* output_buffer = output.buffer();
            UpdateDynamicCircularBufferAddress(program, cb_in, *a_buffer);
            UpdateDynamicCircularBufferAddress(program, cb_out, *output_buffer);

            std::vector<std::vector<uint32_t>> reader_runtime_args = {input_cores.size(), {0}};  // (num_tiles_per_core)
            std::vector<std::vector<uint32_t>> writer_runtime_args = {input_cores.size(), {0}};  // (num_tiles_per_core)
            std::vector<std::vector<uint32_t>> compute_runtime_args = {
                input_cores.size(), {0}};  // (num_tiles_per_core)

            for (uint32_t i = 0; i < input_cores.size(); i++) {
                const CoreCoord& core = input_cores.at(i);
                reader_runtime_args[i][0] = total_tiles_per_core;
                writer_runtime_args[i][0] = total_tiles_per_core;
                compute_runtime_args[i][0] = total_tiles_per_core;
            }
            SetRuntimeArgs(program, reader_kernel_id, input_cores, reader_runtime_args);
            SetRuntimeArgs(program, writer_kernel_id, input_cores, writer_runtime_args);
            SetRuntimeArgs(program, compute_kernel_id, input_cores, compute_runtime_args);
        };
    set_runtime_args(program, a, output);

    auto override_runtime_arguments_callback = [set_runtime_args](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& output_tensor = output_tensors.at(0);
        set_runtime_args(program, input_tensors.at(0), output_tensor);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::cnn::detail
