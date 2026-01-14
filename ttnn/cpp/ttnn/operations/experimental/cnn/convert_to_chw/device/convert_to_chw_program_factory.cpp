// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_chw_program_factory.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::cnn::to_chw::program {

using namespace tt::constants;

namespace {
// Helper function to set runtime arguments for reader, writer, and compute kernels
void set_runtime_args_for_all_kernels(
    tt::tt_metal::Program& program,
    const std::vector<CoreCoord>& cores,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    tt::tt_metal::KernelHandle compute_kernel_id,
    uint32_t total_tiles_per_core) {
    std::vector<std::vector<uint32_t>> reader_runtime_args = {cores.size(), {0}};   // (num_tiles_per_core)
    std::vector<std::vector<uint32_t>> writer_runtime_args = {cores.size(), {0}};   // (num_tiles_per_core)
    std::vector<std::vector<uint32_t>> compute_runtime_args = {cores.size(), {0}};  // (num_tiles_per_core)

    for (uint32_t i = 0; i < cores.size(); i++) {
        reader_runtime_args[i][0] = total_tiles_per_core;
        writer_runtime_args[i][0] = total_tiles_per_core;
        compute_runtime_args[i][0] = total_tiles_per_core;
    }
    SetRuntimeArgs(program, reader_kernel_id, cores, reader_runtime_args);
    SetRuntimeArgs(program, writer_kernel_id, cores, writer_runtime_args);
    SetRuntimeArgs(program, compute_kernel_id, cores, compute_runtime_args);
}
}  // namespace

ConvertToCHWProgramFactory::cached_program_t ConvertToCHWProgramFactory::create(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const auto& a = tensor_args.input;
    auto& output = tensor_return_value;

    const auto& input_shape = a.logical_shape();
    const auto input_core_grid = a.shard_spec()->grid;
    const auto input_cores = corerange_to_cores(
        input_core_grid, std::nullopt, a.shard_spec()->orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);

    const auto output_shard_shape = output.shard_spec()->shape;

    const auto HW = input_shape[2];
    const auto C = input_shape[3];

    log_debug(tt::LogType::LogOp, "Running op with HW={}, C={}, shard_shape={}", HW, C, a.shard_spec()->shape);

    TT_FATAL(C <= TILE_HEIGHT, "C must not exceed 32");
    TT_FATAL(
        tt::div_up(HW, a.shard_spec()->shape[0]) == input_cores.size(),
        "Mismatch between core grid and input/shard shapes");

    const uint32_t total_tiles = HW / TILE_HEIGHT;  // assume C < 32
    const uint32_t total_tiles_per_core = tt::div_up(total_tiles, input_cores.size());

    log_debug(tt::LogType::LogOp, "Processing {} tiles per core ({} total tiles)", total_tiles_per_core, total_tiles);

    const auto create_circular_buffer = [&program, &input_core_grid](
                                            uint32_t index,
                                            uint32_t total_size,
                                            uint32_t page_size,
                                            const tt::DataFormat& format,
                                            tt::tt_metal::Buffer* buffer = nullptr) -> tt::tt_metal::CBHandle {
        log_debug(
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

    const tt::DataFormat input_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    const uint32_t input_tile_size = tt::tile_size(input_format);

    const tt::DataFormat intermediary_format = tt::DataFormat::Float16_b;
    const uint32_t intermediary_tile_size = tt::tile_size(intermediary_format);

    const uint32_t cb_in_id = tt::CBIndex::c_0;
    const uint32_t cb_in_total_size = total_tiles_per_core * input_tile_size;
    const uint32_t cb_in_page_size = input_tile_size;
    const auto cb_in = create_circular_buffer(cb_in_id, cb_in_total_size, cb_in_page_size, input_format, a.buffer());

    const tt::DataFormat output_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t cb_out_id = tt::CBIndex::c_1;
    const uint32_t element_size = tt::datum_size(output_format);
    const uint32_t cb_out_total_size = output_shard_shape[0] * output_shard_shape[1] * element_size;
    const uint32_t cb_out_page_size = output_shard_shape[1] * element_size;
    const auto cb_out =
        create_circular_buffer(cb_out_id, cb_out_total_size, cb_out_page_size, output_format, output.buffer());

    const uint32_t cb_in_transpose_id = tt::CBIndex::c_2;
    const uint32_t cb_in_transpose_total_size = 16 * intermediary_tile_size;
    const uint32_t cb_in_transpose_page_size = intermediary_tile_size;
    create_circular_buffer(
        cb_in_transpose_id, cb_in_transpose_total_size, cb_in_transpose_page_size, intermediary_format);

    std::vector<uint32_t> reader_compile_time_args = {cb_in_id};
    std::vector<uint32_t> writer_compile_time_args = {cb_in_transpose_id, cb_out_id, C};
    std::vector<uint32_t> compute_compile_time_args = {cb_in_id, cb_in_transpose_id};

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

    // Set initial runtime args
    tt::tt_metal::Buffer* a_buffer = a.buffer();
    tt::tt_metal::Buffer* output_buffer = output.buffer();
    UpdateDynamicCircularBufferAddress(program, cb_in, *a_buffer);
    UpdateDynamicCircularBufferAddress(program, cb_out, *output_buffer);

    set_runtime_args_for_all_kernels(
        program, input_cores, reader_kernel_id, writer_kernel_id, compute_kernel_id, total_tiles_per_core);

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .cb_in = cb_in,
            .cb_out = cb_out,
            .input_cores = input_cores,
            .reader_kernel_id = reader_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .compute_kernel_id = compute_kernel_id,
            .total_tiles_per_core = total_tiles_per_core,
        }};
}

void ConvertToCHWProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& output) {
    auto& program = cached_program.program;
    const auto& cb_in = cached_program.shared_variables.cb_in;
    const auto& cb_out = cached_program.shared_variables.cb_out;
    const auto& input_cores = cached_program.shared_variables.input_cores;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& compute_kernel_id = cached_program.shared_variables.compute_kernel_id;
    const auto& total_tiles_per_core = cached_program.shared_variables.total_tiles_per_core;

    // buffers are not provided so we take them from output/input tensors
    auto* output_dram_buffer = output.buffer();
    auto* input_dram_buffer = tensor_args.input.buffer();

    UpdateDynamicCircularBufferAddress(program, cb_in, *input_dram_buffer);
    UpdateDynamicCircularBufferAddress(program, cb_out, *output_dram_buffer);

    set_runtime_args_for_all_kernels(
        program, input_cores, reader_kernel_id, writer_kernel_id, compute_kernel_id, total_tiles_per_core);
}

}  // namespace ttnn::operations::experimental::cnn::to_chw::program
