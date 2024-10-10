// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prefix_scan_program_factory.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::ssm::detail {

using namespace tt::constants;

operation::ProgramWithCallbacks multi_core_ssm_prefix_scan(
    const Tensor& a,
    const Tensor& bx,
    const Tensor& h,
    Tensor& output,
    MathFidelity math_fidelity,
    CoreCoord compute_with_storage_grid_size) {
    auto program = tt::tt_metal::CreateProgram();

    auto* a_buffer = a.buffer();
    auto* bx_buffer = bx.buffer();
    auto* h_buffer = h.buffer();
    auto* output_buffer = output.buffer();
    TT_ASSERT(output_buffer != nullptr, "Output buffer should be allocated on device");

    const tt::DataFormat input_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    const uint32_t input_tile_size = tt::tt_metal::detail::TileSize(input_format);

    const tt::DataFormat intermediary_format = tt::DataFormat::Float16_b;
    const uint32_t intermediary_tile_size = tt::tt_metal::detail::TileSize(intermediary_format);

    const auto all_cores = a.shard_spec()->grid;
    const auto create_circular_buffer = [&program, &all_cores](
                                            uint32_t index,
                                            uint32_t num_tiles,
                                            uint32_t tile_size,
                                            const tt::DataFormat& format,
                                            Buffer* buffer = nullptr) -> tt::tt_metal::CBHandle {
        auto config = CircularBufferConfig(num_tiles * tile_size, {{index, format}}).set_page_size(index, tile_size);
        if (buffer != nullptr) {
            config = config.set_globally_allocated_address(*buffer);
        }
        return tt::tt_metal::CreateCircularBuffer(program, all_cores, config);
    };

    const uint32_t sharded_sequence_length = a.shard_spec()->shape[0];
    const uint32_t sharded_hidden_state_length = a.shard_spec()->shape[1];

    const uint32_t total_tiles_per_row = sharded_hidden_state_length / TILE_HEIGHT;
    const uint32_t total_tiles_per_col = sharded_sequence_length / TILE_HEIGHT;
    const uint32_t total_tiles = total_tiles_per_row * total_tiles_per_col;

    // One chunk is a row of 32 tiles where an untilize call will move each row into a seperate tile
    const uint32_t num_tiles_in_chunk = 32;
    const uint32_t num_chunks_per_row = tt::div_up(total_tiles_per_row, num_tiles_in_chunk);

    const uint32_t cb_a_in_id = tt::CB::c_in0;
    const auto cb_a_in = create_circular_buffer(cb_a_in_id, total_tiles, input_tile_size, input_format, a_buffer);

    const uint32_t cb_bx_in_id = tt::CB::c_in1;
    const auto cb_bx_in = create_circular_buffer(cb_bx_in_id, total_tiles, input_tile_size, input_format, bx_buffer);

    // Hidden state is in row-major so must be bfloat16
    const uint32_t cb_h_in_id = tt::CB::c_in2;
    const auto cb_h_in =
        create_circular_buffer(cb_h_in_id, num_chunks_per_row, intermediary_tile_size, intermediary_format, h_buffer);

    const uint32_t cb_out_id = tt::CB::c_out0;
    const auto cb_out = create_circular_buffer(cb_out_id, total_tiles, input_tile_size, input_format, output_buffer);

    const uint32_t num_tiles_in_row_to_tile_cb = 32;  // Tilizing 32 tiles will pack tensor rows into seperate tiles
    const uint32_t cb_a_tilize_in_id = tt::CB::c_intermed0;
    const auto cb_a_tilize_in = create_circular_buffer(
        cb_a_tilize_in_id, num_tiles_in_row_to_tile_cb, intermediary_tile_size, intermediary_format);

    const uint32_t cb_bx_tilize_in_id = tt::CB::c_intermed1;
    const auto cb_b_tilize_in = create_circular_buffer(
        cb_bx_tilize_in_id, num_tiles_in_row_to_tile_cb, intermediary_tile_size, intermediary_format);

    const uint32_t cb_tilize_out_id = tt::CB::c_intermed2;
    const auto cb_tilize_out = create_circular_buffer(
        cb_tilize_out_id, num_tiles_in_row_to_tile_cb, intermediary_tile_size, intermediary_format);

    const uint32_t cb_h_prev_id = tt::CB::c_intermed3;
    const auto cb_h_prev = create_circular_buffer(cb_h_prev_id, 2, intermediary_tile_size, intermediary_format);

    const uint32_t cb_ah_id = tt::CB::c_intermed4;
    const auto cb_ah = create_circular_buffer(cb_ah_id, 2, intermediary_tile_size, intermediary_format);

    const uint32_t cb_h_id = tt::CB::c_intermed5;
    const auto cb_h = create_circular_buffer(cb_h_id, 2, intermediary_tile_size, intermediary_format);

    const uint32_t cb_h_acc_id = tt::CB::c_intermed7;
    const auto cb_h_acc =
        create_circular_buffer(cb_h_acc_id, num_chunks_per_row, intermediary_tile_size, intermediary_format);

    std::vector<uint32_t> reader_compile_time_args = {cb_a_in_id, cb_bx_in_id, cb_h_in_id};
    std::vector<uint32_t> writer_compile_time_args = {cb_out_id, cb_h_acc_id, cb_h_in_id};
    std::vector<uint32_t> compute_compile_time_args = {
        cb_a_in_id,
        cb_bx_in_id,
        cb_h_in_id,
        cb_a_tilize_in_id,
        cb_bx_tilize_in_id,
        cb_h_prev_id,
        cb_ah_id,
        cb_h_id,
        cb_tilize_out_id,
        cb_out_id,
        cb_h_acc_id};

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ssm/prefix_scan/device/kernels/reader_ssm_prefix_scan.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ssm/prefix_scan/device/kernels/writer_ssm_prefix_scan.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ssm/prefix_scan/device/kernels/ssm_prefix_scan.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args});

    std::vector<CoreCoord> cores =
        grid_to_cores(all_cores.num_cores(), compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, true);

    auto set_runtime_args =
        [reader_kernel_id,
         writer_kernel_id,
         compute_kernel_id,
         total_tiles,
         total_tiles_per_row,
         total_tiles_per_col,
         num_chunks_per_row,
         sharded_hidden_state_length,
         all_cores,
         cores,
         cb_a_in,
         cb_bx_in,
         cb_h_in,
         cb_out](ProgramHandle program, const Tensor& a, const Tensor& bx, const Tensor& h, const Tensor& output) {
            tt::tt_metal::Buffer* a_buffer = a.buffer();
            tt::tt_metal::Buffer* bx_buffer = bx.buffer();
            tt::tt_metal::Buffer* h_buffer = h.buffer();
            tt::tt_metal::Buffer* output_buffer = output.buffer();

            UpdateDynamicCircularBufferAddress(program, cb_a_in, *a_buffer);
            UpdateDynamicCircularBufferAddress(program, cb_bx_in, *bx_buffer);
            UpdateDynamicCircularBufferAddress(program, cb_h_in, *h_buffer);
            UpdateDynamicCircularBufferAddress(program, cb_out, *output_buffer);

            std::vector<std::vector<uint32_t>> reader_runtime_args = {
                cores.size(), {0, 0}};  // (num_tiles_per_core, num_chunks_per_row)
            std::vector<std::vector<uint32_t>> writer_runtime_args = {
                cores.size(), {0, 0}};  // (num_tiles_per_core, hidden_state_len)
            std::vector<std::vector<uint32_t>> compute_runtime_args = {
                cores.size(),
                {0, 0, 0, 0}};  // (total_tiles, total_tiles_per_row, total_tiles_per_col, num_chunks_per_row)

            for (uint32_t i = 0, num_blocks_written = 0; i < cores.size(); i++) {
                const CoreCoord& core = cores.at(i);

                reader_runtime_args[i][0] = total_tiles;
                reader_runtime_args[i][1] = num_chunks_per_row;

                writer_runtime_args[i][0] = total_tiles;
                writer_runtime_args[i][1] = sharded_hidden_state_length;

                compute_runtime_args[i][0] = total_tiles;
                compute_runtime_args[i][1] = total_tiles_per_row;
                compute_runtime_args[i][2] = total_tiles_per_col;
                compute_runtime_args[i][3] = num_chunks_per_row;
            }
            SetRuntimeArgs(program, reader_kernel_id, cores, reader_runtime_args);
            SetRuntimeArgs(program, writer_kernel_id, cores, writer_runtime_args);
            SetRuntimeArgs(program, compute_kernel_id, cores, compute_runtime_args);
        };

    set_runtime_args(program, a, bx, h, output);

    auto override_runtime_arguments_callback = [set_runtime_args](
                                                   const void* operation,
                                                   ProgramHandle program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto& a = input_tensors.at(0);
        auto& bx = input_tensors.at(1);
        auto& h = input_tensors.at(2);
        auto& out = output_tensors.at(0);
        set_runtime_args(program, a, bx, h, out);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::ssm::detail
