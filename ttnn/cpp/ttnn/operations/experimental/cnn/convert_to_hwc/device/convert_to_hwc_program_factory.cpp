// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_hwc_program_factory.hpp"

#include "tt-metalium/tt_backend_api_types.hpp"

#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"

namespace ttnn::operations::experimental::cnn::detail {

using namespace tt::constants;

tt::tt_metal::operation::ProgramWithCallbacks multi_core_convert_to_hwc(const Tensor& a, Tensor& output) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // If we are pulling from DRAM we infer the shard shape by using the output shard spec
    const bool is_input_in_dram = a.buffer()->core_type() == CoreType::DRAM;
    const uint32_t input_shard_height = is_input_in_dram ? output.shard_spec()->shape[1] : a.shard_spec()->shape[0];
    const uint32_t input_shard_width = is_input_in_dram ? output.shard_spec()->shape[0] : a.shard_spec()->shape[1];
    const CoreRangeSet input_core_grid = is_input_in_dram ? output.shard_spec()->grid : a.shard_spec()->grid;
    const std::vector<CoreCoord> input_cores = corerange_to_cores(
        input_core_grid, std::nullopt, a.shard_spec()->orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);

    TT_FATAL(input_shard_height <= TILE_HEIGHT, "Shard height must be 32 or smaller");
    TT_FATAL(input_shard_height % 8 == 0, "Shard height must be mutliple of 8");
    TT_FATAL(input_shard_width % TILE_WIDTH == 0, "Shard width must be multiple of tile width");

    const auto create_circular_buffer = [&program, &input_core_grid](
                                            uint32_t index,
                                            uint32_t total_size,
                                            uint32_t page_size,
                                            const tt::DataFormat& format,
                                            tt::tt_metal::Buffer* buffer = nullptr) -> tt::tt_metal::CBHandle {
        auto config = tt::tt_metal::CircularBufferConfig(total_size, {{index, format}}).set_page_size(index, page_size);
        if (buffer != nullptr) {
            config = config.set_globally_allocated_address(*buffer);
        }
        return tt::tt_metal::CreateCircularBuffer(program, input_core_grid, config);
    };

    const tt::DataFormat intermediary_format = tt::DataFormat::Float16_b;
    const uint32_t intermediary_tile_size = tt::tt_metal::detail::TileSize(intermediary_format);

    const uint32_t cb_in_id = tt::CBIndex::c_0;
    const tt::DataFormat input_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    const uint32_t input_element_size = tt::datum_size(input_format);
    const uint32_t cb_in_page_size = input_shard_width * input_element_size;
    const uint32_t cb_in_total_size = input_shard_height * cb_in_page_size;
    const auto cb_in = create_circular_buffer(
        cb_in_id, cb_in_total_size, cb_in_page_size, input_format, is_input_in_dram ? nullptr : a.buffer());

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

    const uint32_t total_tiles_per_core = tt::div_up(input_shard_width, TILE_HEIGHT);
    const uint32_t total_tiles_writer0 = tt::div_up(total_tiles_per_core, 2);
    const uint32_t total_tiles_writer1 = total_tiles_per_core - total_tiles_writer0;
    uint32_t output_stride_sticks = TILE_WIDTH;  // needed to stride output address when doing split writers

    const auto dram_input_cores = corerange_to_cores(
        a.shard_spec()->grid, std::nullopt, a.shard_spec()->orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
    const auto remote_core_type = a.buffer()->core_type();
    const auto remote_address = a.buffer()->address();
    const auto remote_buffer_type = a.buffer()->buffer_type();
    const auto [runtime_args_for_each_core, _, dram_write_stride_bytes, dram_read_stride_bytes] =
        data_movement::detail::compute_width_sharding_reshard_segments(
            {input_shard_height, input_shard_width},
            a.shard_spec()->shape,  // dram shard shape
            input_cores,
            dram_input_cores,
            remote_buffer_type,
            remote_core_type,
            a.device(),
            2);

    // Split DRAM read across each kernel along tensor height since this is the best way to split work evenly
    const uint32_t total_num_sticks_kernel_0 = input_shard_height / 2;
    const uint32_t total_num_sticks_kernel_1 = input_shard_height - total_num_sticks_kernel_0;

    std::vector<uint32_t> writer_compile_time_args0 = {
        cb_in_id,
        cb_in_transpose_id0,
        cb_out_id,
        input_shard_height,
        input_shard_width,
        total_tiles_writer0,
        output_stride_sticks,
        0,
        input_element_size,
        is_input_in_dram,
        true,
        dram_write_stride_bytes,
        dram_read_stride_bytes,
        total_num_sticks_kernel_0};

    std::vector<uint32_t> writer_compile_time_args1 = {
        cb_in_id,
        cb_in_transpose_id1,
        cb_out_id,
        input_shard_height,
        input_shard_width,
        total_tiles_writer1,
        output_stride_sticks,
        output_stride_sticks,
        input_element_size,
        is_input_in_dram,
        false,
        dram_write_stride_bytes,
        dram_read_stride_bytes,
        total_num_sticks_kernel_1};

    std::vector<uint32_t> compute_compile_time_args = {
        cb_in_id,
        cb_in_tiled_id,
        cb_in_transpose_id0,
        cb_in_transpose_id1,
        total_tiles_per_core,
        input_shard_height,
        is_input_in_dram};

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

    auto set_runtime_args = [cb_in,
                             cb_out,
                             is_input_in_dram,
                             input_cores,
                             runtime_args_for_each_core,
                             writer_kernel_id0,
                             writer_kernel_id1,
                             total_num_sticks_kernel_0,
                             total_num_sticks_kernel_1,
                             remote_address,
                             dram_read_stride_bytes,
                             dram_write_stride_bytes](
                                tt::tt_metal::Program& program, const Tensor& a, const Tensor& output) {
        if (is_input_in_dram) {
            for (uint32_t core_idx = 0; core_idx < input_cores.size(); core_idx++) {
                const auto& args_for_all_segments = runtime_args_for_each_core[core_idx];
                std::vector<uint32_t> runtime_args_0 = {remote_address, args_for_all_segments.size()};
                std::vector<uint32_t> runtime_args_1 = {remote_address, args_for_all_segments.size()};
                for (const auto& args : args_for_all_segments) {
                    const std::vector<uint32_t> segment_kernel_0 = {
                        args.write_size, args.read_offset, args.bank_id, args.write_offset};
                    runtime_args_0.insert(runtime_args_0.end(), segment_kernel_0.begin(), segment_kernel_0.end());

                    // Adjust read and write offsets to the correct stick address because we are splitting work across 2
                    // kernels
                    const uint32_t adjusted_read_offset =
                        args.read_offset + total_num_sticks_kernel_0 * dram_write_stride_bytes;
                    const uint32_t adjusted_write_offset =
                        args.write_offset + total_num_sticks_kernel_0 * dram_read_stride_bytes;

                    const std::vector<uint32_t> segment_kernel_1 = {
                        args.write_size, adjusted_read_offset, args.bank_id, adjusted_write_offset};
                    runtime_args_1.insert(runtime_args_1.end(), segment_kernel_1.begin(), segment_kernel_1.end());
                }
                SetRuntimeArgs(program, writer_kernel_id0, input_cores[core_idx], runtime_args_0);
                SetRuntimeArgs(program, writer_kernel_id1, input_cores[core_idx], runtime_args_1);
            }
        } else {
            tt::tt_metal::Buffer* a_buffer = a.buffer();
            UpdateDynamicCircularBufferAddress(program, cb_in, *a_buffer);
        }
        tt::tt_metal::Buffer* output_buffer = output.buffer();
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
}

}  // namespace ttnn::operations::experimental::cnn::detail
