// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prefix_scan_program_factory.hpp"

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::ssm::prefix_scan::program {

using namespace tt::constants;

PrefixScanProgramFactory::cached_program_t PrefixScanProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& tensor_return_value) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const auto& a = tensor_args.a;
    const auto& bx = tensor_args.bx;
    const auto& h_prev = tensor_args.h_prev;
    auto& output = tensor_return_value;

    auto* a_buffer = a.buffer();
    auto* bx_buffer = bx.buffer();
    auto* h_buffer = h_prev.buffer();
    auto* output_buffer = output.buffer();
    TT_ASSERT(output_buffer != nullptr, "Output buffer should be allocated on device");

    const tt::DataFormat input_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    const uint32_t input_tile_size = tt::tile_size(input_format);

    const tt::DataFormat intermediary_format = tt::DataFormat::Float16_b;
    const uint32_t intermediary_row_size = tt::datum_size(intermediary_format) * TILE_WIDTH;
    const uint32_t intermediary_tile_size = tt::tile_size(intermediary_format);

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

    // One chunk is a row of 32 tiles where an untilize call will move each row into a separate tile
    constexpr uint32_t num_tiles_in_chunk = 32;
    const uint32_t num_chunks_per_row = tt::div_up(total_tiles_per_row, num_tiles_in_chunk);

    const uint32_t cb_a_in_id = tt::CBIndex::c_0;
    const auto cb_a_in = create_circular_buffer(cb_a_in_id, total_tiles, input_tile_size, input_format, a_buffer);

    const uint32_t cb_bx_in_id = tt::CBIndex::c_1;
    const auto cb_bx_in = create_circular_buffer(cb_bx_in_id, total_tiles, input_tile_size, input_format, bx_buffer);

    // Hidden state is in row-major so must be bfloat16
    const uint32_t cb_h_in_id = tt::CBIndex::c_2;
    const auto cb_h_in =
        create_circular_buffer(cb_h_in_id, total_tiles_per_row, intermediary_row_size, intermediary_format, h_buffer);

    const uint32_t cb_out_id = tt::CBIndex::c_16;
    const auto cb_out = create_circular_buffer(cb_out_id, total_tiles, input_tile_size, input_format, output_buffer);

    const uint32_t num_tiles_in_row_to_tile_cb = 32;  // Tilizing 32 tiles will pack tensor rows into separate tiles
    const uint32_t cb_a_tilize_in_id = tt::CBIndex::c_24;
    create_circular_buffer(cb_a_tilize_in_id, num_tiles_in_row_to_tile_cb, intermediary_tile_size, intermediary_format);

    const uint32_t cb_bx_tilize_in_id = tt::CBIndex::c_25;
    create_circular_buffer(
        cb_bx_tilize_in_id, num_tiles_in_row_to_tile_cb, intermediary_tile_size, intermediary_format);

    const uint32_t cb_tilize_out_id = tt::CBIndex::c_26;
    create_circular_buffer(cb_tilize_out_id, num_tiles_in_row_to_tile_cb, intermediary_tile_size, intermediary_format);

    const uint32_t cb_h_prev_id = tt::CBIndex::c_27;
    create_circular_buffer(cb_h_prev_id, 2, intermediary_tile_size, intermediary_format);

    const uint32_t cb_ah_id = tt::CBIndex::c_28;
    create_circular_buffer(cb_ah_id, 2, intermediary_tile_size, intermediary_format);

    const uint32_t cb_h_id = tt::CBIndex::c_29;
    create_circular_buffer(cb_h_id, 2, intermediary_tile_size, intermediary_format);

    const uint32_t cb_h_acc_id = tt::CBIndex::c_31;
    create_circular_buffer(cb_h_acc_id, num_chunks_per_row, intermediary_tile_size, intermediary_format);

    std::vector<uint32_t> reader_compile_time_args = {cb_a_in_id, cb_bx_in_id, cb_h_in_id};
    std::vector<uint32_t> writer_compile_time_args = {cb_out_id, cb_h_acc_id, cb_h_in_id};
    tt::tt_metal::TensorAccessorArgs(a_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(bx_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(h_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
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
            .math_fidelity = operation_attributes.math_fidelity,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args});

    auto device_compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();
    std::vector<CoreCoord> cores = grid_to_cores(
        all_cores.num_cores(), device_compute_with_storage_grid_size.x, device_compute_with_storage_grid_size.y, true);

    // Store shared variables
    PrefixScanSharedVariables shared_variables;
    shared_variables.reader_kernel_id = reader_kernel_id;
    shared_variables.writer_kernel_id = writer_kernel_id;
    shared_variables.compute_kernel_id = compute_kernel_id;
    shared_variables.cores = cores;
    shared_variables.cb_a_in = cb_a_in;
    shared_variables.cb_bx_in = cb_bx_in;
    shared_variables.cb_h_in = cb_h_in;
    shared_variables.cb_out = cb_out;
    shared_variables.total_tiles = total_tiles;
    shared_variables.total_tiles_per_row = total_tiles_per_row;
    shared_variables.total_tiles_per_col = total_tiles_per_col;
    shared_variables.num_chunks_per_row = num_chunks_per_row;
    shared_variables.sharded_hidden_state_length = sharded_hidden_state_length;

    cached_program_t cached_program{std::move(program), std::move(shared_variables)};

    // Set initial runtime args
    override_runtime_arguments(cached_program, operation_attributes, tensor_args, tensor_return_value);

    return cached_program;
}

void PrefixScanProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    const auto& a = tensor_args.a;
    const auto& bx = tensor_args.bx;
    const auto& h_prev = tensor_args.h_prev;
    auto& output = tensor_return_value;

    tt::tt_metal::Buffer* a_buffer = a.buffer();
    tt::tt_metal::Buffer* bx_buffer = bx.buffer();
    tt::tt_metal::Buffer* h_buffer = h_prev.buffer();
    tt::tt_metal::Buffer* output_buffer = output.buffer();

    auto& program = cached_program.program;
    const auto& shared_vars = cached_program.shared_variables;

    UpdateDynamicCircularBufferAddress(program, shared_vars.cb_a_in, *a_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_vars.cb_bx_in, *bx_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_vars.cb_h_in, *h_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_vars.cb_out, *output_buffer);

    std::vector<std::vector<uint32_t>> reader_runtime_args = {
        shared_vars.cores.size(), {0, 0}};  // (num_tiles_per_core, total_tiles_per_row)
    std::vector<std::vector<uint32_t>> writer_runtime_args = {
        shared_vars.cores.size(), {0, 0}};  // (num_tiles_per_core, hidden_state_len)
    std::vector<std::vector<uint32_t>> compute_runtime_args = {
        shared_vars.cores.size(),
        {0, 0, 0, 0}};  // (total_tiles, total_tiles_per_row, total_tiles_per_col, num_chunks_per_row)

    for (uint32_t i = 0; i < shared_vars.cores.size(); i++) {
        reader_runtime_args[i][0] = shared_vars.total_tiles;
        reader_runtime_args[i][1] = shared_vars.total_tiles_per_row;

        writer_runtime_args[i][0] = shared_vars.total_tiles;
        writer_runtime_args[i][1] = shared_vars.sharded_hidden_state_length;

        compute_runtime_args[i][0] = shared_vars.total_tiles;
        compute_runtime_args[i][1] = shared_vars.total_tiles_per_row;
        compute_runtime_args[i][2] = shared_vars.total_tiles_per_col;
        compute_runtime_args[i][3] = shared_vars.num_chunks_per_row;
    }
    SetRuntimeArgs(program, shared_vars.reader_kernel_id, shared_vars.cores, reader_runtime_args);
    SetRuntimeArgs(program, shared_vars.writer_kernel_id, shared_vars.cores, writer_runtime_args);
    SetRuntimeArgs(program, shared_vars.compute_kernel_id, shared_vars.cores, compute_runtime_args);
}

}  // namespace ttnn::operations::experimental::ssm::prefix_scan::program
