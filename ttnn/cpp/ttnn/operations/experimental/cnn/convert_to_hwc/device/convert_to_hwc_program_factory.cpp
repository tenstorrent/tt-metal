// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_hwc_program_factory.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::cnn::detail {

using namespace tt::constants;

tt::tt_metal::operation::ProgramWithCallbacks multi_core_convert_to_hwc(const Tensor& a, Tensor& output) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const auto input_shape = a.get_logical_shape();
    const auto input_shard_height = a.shard_spec()->shape[0];
    const auto input_shard_width = a.shard_spec()->shape[1];
    const auto input_core_grid = a.shard_spec()->grid;
    const auto input_cores = corerange_to_cores(
        input_core_grid, std::nullopt, a.shard_spec()->orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);

    const auto C = input_shape[2];
    const auto HW = input_shape[3];

    tt::log_debug(tt::LogType::LogOp, "Running op with C={}, HW={}, shard_shape={}", C, HW, a.shard_spec()->shape);

    TT_FATAL(input_shard_height <= TILE_HEIGHT, "Shard height must be 32 or smaller");
    TT_FATAL(input_shard_width % TILE_WIDTH == 0, "Shard width must be multiple of tile width");

    const uint32_t total_tiles_per_core = tt::div_up(input_shard_width, TILE_HEIGHT);

    tt::log_debug(tt::LogType::LogOp, "Processing {} tiles per core", total_tiles_per_core);

    const auto create_circular_buffer = [&program, &input_core_grid](
                                            uint32_t index,
                                            uint32_t total_size,
                                            uint32_t page_size,
                                            const tt::DataFormat& format,
                                            tt::tt_metal::Buffer* buffer = nullptr) -> tt::tt_metal::CBHandle {
        tt::log_debug(
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
    const uint32_t cb_in_page_size = input_shard_width * input_element_size;
    const uint32_t cb_in_total_size = input_shard_height * cb_in_page_size;
    const auto cb_in = create_circular_buffer(cb_in_id, cb_in_total_size, cb_in_page_size, input_format, a.buffer());

    const uint32_t cb_in_tiled_id = tt::CBIndex::c_1;
    const uint32_t cb_in_tiled_total_size = tt::div_up(input_shard_width, TILE_WIDTH) * intermediary_tile_size;
    const uint32_t cb_in_tiled_page_size = intermediary_tile_size;
    const auto cb_in_tiled =
        create_circular_buffer(cb_in_tiled_id, cb_in_tiled_total_size, cb_in_tiled_page_size, intermediary_format);

    const uint32_t cb_in_transpose_total_size = tt::div_up(input_shard_width, TILE_WIDTH) * intermediary_tile_size;
    const uint32_t cb_in_transpose_page_size = intermediary_tile_size;

    const uint32_t cb_in_transpose_id0 = tt::CBIndex::c_2;
    const auto cb_in_transpose0 = create_circular_buffer(
        cb_in_transpose_id0, cb_in_transpose_total_size, cb_in_transpose_page_size, intermediary_format);

    const uint32_t cb_in_transpose_id1 = tt::CBIndex::c_3;
    const auto cb_in_transpose1 = create_circular_buffer(
        cb_in_transpose_id1, cb_in_transpose_total_size, cb_in_transpose_page_size, intermediary_format);

    const uint32_t cb_out_id = tt::CBIndex::c_4;
    const tt::DataFormat output_format = input_format;
    const uint32_t cb_out_total_size = cb_in_total_size;  // same size as input
    const uint32_t cb_out_page_size = input_shard_height * input_element_size;
    const auto cb_out =
        create_circular_buffer(cb_out_id, cb_out_total_size, cb_out_page_size, output_format, output.buffer());

    // Divide work between data movement cores
    const uint32_t total_tiles_writer0 = tt::div_up(total_tiles_per_core, 2);
    const uint32_t total_tiles_writer1 = total_tiles_per_core - total_tiles_writer0;
    uint32_t output_stride_sticks = TILE_WIDTH;  // needed to stride output address when doing split writers

    std::vector<uint32_t> writer_compile_time_args0 = {
        cb_in_transpose_id0, cb_out_id, C, input_shard_width, total_tiles_writer0, output_stride_sticks, 0};
    std::vector<uint32_t> writer_compile_time_args1 = {
        cb_in_transpose_id1,
        cb_out_id,
        C,
        input_shard_width,
        total_tiles_writer1,
        output_stride_sticks,
        output_stride_sticks};
    std::vector<uint32_t> compute_compile_time_args = {
        cb_in_id, cb_in_tiled_id, cb_in_transpose_id0, cb_in_transpose_id1, total_tiles_per_core, input_shard_height};

    auto writer_kernel_id0 = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/kernels/writer_convert_to_hwc.cpp",
        input_core_grid,
        tt::tt_metal::ReaderDataMovementConfig(writer_compile_time_args0));

    auto writer_kernel_id1 = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/kernels/writer_convert_to_hwc.cpp",
        input_core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args1));

    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/kernels/convert_to_hwc.cpp",
        input_core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args});

    auto set_runtime_args = [cb_in, cb_out](tt::tt_metal::Program& program, const Tensor& a, const Tensor& output) {
        tt::tt_metal::Buffer* a_buffer = a.buffer();
        tt::tt_metal::Buffer* output_buffer = output.buffer();
        UpdateDynamicCircularBufferAddress(program, cb_in, *a_buffer);
        UpdateDynamicCircularBufferAddress(program, cb_out, *output_buffer);
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
}  // namespace ttnn::operations::experimental::cnn::detail

}  // namespace ttnn::operations::experimental::cnn::detail
