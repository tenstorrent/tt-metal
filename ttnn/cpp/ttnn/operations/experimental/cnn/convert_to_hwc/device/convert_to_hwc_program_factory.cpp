// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_hwc_program_factory.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::cnn::detail {

using namespace tt::constants;

tt::tt_metal::operation::ProgramWithCallbacks multi_core_convert_to_hwc(const Tensor& a, Tensor& output) {
    tt::log_info("STARTING CONVERT OP");
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const auto input_shape = a.get_logical_shape();
    const auto input_shard_height = a.shard_spec()->shape[0];
    const auto input_shard_width = a.shard_spec()->shape[1];
    const auto input_core_grid = a.shard_spec()->grid;
    const auto input_cores = corerange_to_cores(
        input_core_grid, std::nullopt, a.shard_spec()->orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
    tt::log_info("STARTING CONVERT OP");

    const auto C = input_shape[2];
    const auto HW = input_shape[3];

    tt::log_info(tt::LogType::LogOp, "Running op with C={}, HW={}, shard_shape={}", C, HW, a.shard_spec()->shape);
    tt::log_info("STARTING CONVERT OP");

    TT_FATAL(C < TILE_HEIGHT, "HW must be 32 or smaller");

    const uint32_t total_tiles = HW / TILE_HEIGHT;  // assume HW < 32
    const uint32_t total_tiles_per_core = tt::div_up(total_tiles, input_cores.size());
    tt::log_info("STARTING CONVERT OP");

    tt::log_info(
        tt::LogType::LogOp, "Processing {} tiles per core ({} total tiles)", total_tiles_per_core, total_tiles);

    const auto create_circular_buffer = [&program, &input_core_grid](
                                            uint32_t index,
                                            uint32_t total_size,
                                            uint32_t page_size,
                                            const tt::DataFormat& format,
                                            tt::tt_metal::Buffer* buffer = nullptr) -> tt::tt_metal::CBHandle {
        tt::log_info(
            tt::LogType::LogOp,
            "Creating CB at index {} with total size {} B and page size {} B",
            index,
            total_size,
            page_size);
        auto config = tt::tt_metal::CircularBufferConfig(total_size, {{index, format}}).set_page_size(index, page_size);
        if (buffer != nullptr) {
            config = config.set_globally_allocated_address(*buffer);
        }
        return tt::tt_metal::CreateCircularBuffer(program, input_core_grid, config);
    };

    const tt::DataFormat intermediary_format = tt::DataFormat::Float16_b;
    const uint32_t intermediary_tile_size = tt::tt_metal::detail::TileSize(intermediary_format);

    const uint32_t cb_in_id = tt::CBIndex::c_0;
    const tt::DataFormat input_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    const uint32_t input_element_size = tt::datum_size(input_format);
    const uint32_t cb_in_total_size = input_shard_height * input_shard_width * input_element_size;
    const uint32_t cb_in_page_size = input_shard_width * input_element_size;
    const auto cb_in = create_circular_buffer(cb_in_id, cb_in_total_size, cb_in_page_size, input_format, a.buffer());

    const uint32_t cb_out_id = tt::CBIndex::c_1;
    const tt::DataFormat output_format = input_format;
    const uint32_t cb_out_total_size = cb_in_total_size;  // same size as input
    const uint32_t cb_out_page_size = input_shard_height * input_element_size;
    const auto cb_out =
        create_circular_buffer(cb_out_id, cb_out_total_size, cb_out_page_size, output_format, output.buffer());

    const uint32_t cb_in_tiled_id = tt::CBIndex::c_2;
    const uint32_t cb_in_tiled_total_size = intermediary_tile_size;
    const uint32_t cb_in_tiled_page_size = intermediary_tile_size;
    const auto cb_in_tiled =
        create_circular_buffer(cb_in_tiled_id, cb_in_tiled_total_size, cb_in_tiled_page_size, intermediary_format);

    const uint32_t cb_in_transpose_id = tt::CBIndex::c_3;
    const uint32_t cb_in_transpose_total_size = intermediary_tile_size;
    const uint32_t cb_in_transpose_page_size = intermediary_tile_size;
    const auto cb_in_transpose = create_circular_buffer(
        cb_in_transpose_id, cb_in_transpose_total_size, cb_in_transpose_page_size, intermediary_format);

    std::vector<uint32_t> reader_compile_time_args = {cb_in_id};
    std::vector<uint32_t> writer_compile_time_args = {cb_in_transpose_id, cb_out_id, HW};
    std::vector<uint32_t> compute_compile_time_args = {cb_in_id, cb_in_tiled_id, cb_in_transpose_id};

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/kernels/reader_convert_to_hwc.cpp",
        input_core_grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/kernels/writer_convert_to_hwc.cpp",
        input_core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/kernels/convert_to_hwc.cpp",
        input_core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args});

    auto set_runtime_args =
        [cb_in, cb_out, input_cores, total_tiles_per_core, reader_kernel_id, writer_kernel_id, compute_kernel_id](
            tt::tt_metal::Program& program, const Tensor& a, const Tensor& output) {
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
                                                   tt::tt_metal::Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& output_tensor = output_tensors.at(0);
        set_runtime_args(program, input_tensors.at(0), output_tensor);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::cnn::detail
