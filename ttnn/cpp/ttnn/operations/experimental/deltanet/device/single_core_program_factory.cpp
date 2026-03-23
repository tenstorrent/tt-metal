// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "deltanet_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::deltanet {

DeltaNetRecurrenceOperation::SingleCore::cached_program_t DeltaNetRecurrenceOperation::SingleCore::create(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& conv_out = tensor_args.conv_out;
    const auto& state = tensor_args.state;

    auto* conv_out_buffer = conv_out.buffer();
    auto* state_buffer = state.buffer();
    auto* output_buffer = output_tensor.buffer();

    tt::tt_metal::Program program{};

    // Data format
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(conv_out.dtype());
    uint32_t tile_size = tt::tile_size(cb_data_format);

    // Architecture params
    uint32_t num_heads = attrs.num_heads;               // 48
    uint32_t k_tiles = attrs.head_k_dim / 32;           // 4
    uint32_t v_tiles = attrs.head_v_dim / 32;           // 4
    uint32_t state_tiles_per_head = k_tiles * v_tiles;  // 16

    // Single core
    CoreCoord core = {0, 0};
    CoreRange core_range(core, core);

    // ---- Circular Buffers ----
    // cb_q (c_0): query vector tiles (double-buffered)
    CircularBufferConfig cb_q_config = CircularBufferConfig(2 * tile_size, {{tt::CBIndex::c_0, cb_data_format}})
                                           .set_page_size(tt::CBIndex::c_0, tile_size);
    CreateCircularBuffer(program, core_range, cb_q_config);

    // cb_k (c_1): key vector tiles (double-buffered)
    CircularBufferConfig cb_k_config = CircularBufferConfig(2 * tile_size, {{tt::CBIndex::c_1, cb_data_format}})
                                           .set_page_size(tt::CBIndex::c_1, tile_size);
    CreateCircularBuffer(program, core_range, cb_k_config);

    // cb_v (c_2): value vector tiles (double-buffered)
    CircularBufferConfig cb_v_config = CircularBufferConfig(2 * tile_size, {{tt::CBIndex::c_2, cb_data_format}})
                                           .set_page_size(tt::CBIndex::c_2, tile_size);
    CreateCircularBuffer(program, core_range, cb_v_config);

    // cb_decay (c_3): decay tiles (double-buffered)
    CircularBufferConfig cb_decay_config = CircularBufferConfig(2 * tile_size, {{tt::CBIndex::c_3, cb_data_format}})
                                               .set_page_size(tt::CBIndex::c_3, tile_size);
    CreateCircularBuffer(program, core_range, cb_decay_config);

    // cb_beta (c_4): beta tiles (double-buffered)
    CircularBufferConfig cb_beta_config = CircularBufferConfig(2 * tile_size, {{tt::CBIndex::c_4, cb_data_format}})
                                              .set_page_size(tt::CBIndex::c_4, tile_size);
    CreateCircularBuffer(program, core_range, cb_beta_config);

    // cb_state (c_5): state input tiles (double-buffered)
    CircularBufferConfig cb_state_config = CircularBufferConfig(2 * tile_size, {{tt::CBIndex::c_5, cb_data_format}})
                                               .set_page_size(tt::CBIndex::c_5, tile_size);
    CreateCircularBuffer(program, core_range, cb_state_config);

    // cb_out (c_16): output vector tiles (double-buffered)
    CircularBufferConfig cb_out_config = CircularBufferConfig(2 * tile_size, {{tt::CBIndex::c_16, cb_data_format}})
                                             .set_page_size(tt::CBIndex::c_16, tile_size);
    CreateCircularBuffer(program, core_range, cb_out_config);

    // cb_state_out (c_17): updated state output tiles (double-buffered)
    CircularBufferConfig cb_state_out_config =
        CircularBufferConfig(2 * tile_size, {{tt::CBIndex::c_17, cb_data_format}})
            .set_page_size(tt::CBIndex::c_17, tile_size);
    CreateCircularBuffer(program, core_range, cb_state_out_config);

    // ---- Reader Kernel ----
    // Compile-time args: num_heads, k_tiles, v_tiles, then TensorAccessorArgs for state + conv_out
    std::vector<uint32_t> reader_compile_args = {
        num_heads,
        k_tiles,
        v_tiles,
    };
    TensorAccessorArgs(*state_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(*conv_out_buffer).append_to(reader_compile_args);

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/dataflow/reader.cpp",
        core_range,
        ReaderDataMovementConfig(reader_compile_args));

    // ---- Writer Kernel ----
    // Compile-time args: num_heads, v_tiles, state_tiles_per_head, then TensorAccessorArgs for output + state
    std::vector<uint32_t> writer_compile_args = {
        num_heads,
        v_tiles,
        state_tiles_per_head,
    };
    TensorAccessorArgs(*output_buffer).append_to(writer_compile_args);
    TensorAccessorArgs(*state_buffer).append_to(writer_compile_args);

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/dataflow/writer.cpp",
        core_range,
        WriterDataMovementConfig(writer_compile_args));

    // ---- Compute Kernel ----
    std::vector<uint32_t> compute_compile_args = {
        num_heads,
        k_tiles,
        v_tiles,
    };
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/compute/deltanet_recurrence.cpp",
        core_range,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2, .math_approx_mode = true, .compile_args = compute_compile_args});

    // ---- Runtime Args ----
    SetRuntimeArgs(program, reader_kernel_id, core, {state_buffer->address(), conv_out_buffer->address()});

    SetRuntimeArgs(program, writer_kernel_id, core, {output_buffer->address(), state_buffer->address()});

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .compute_kernel_id = compute_kernel_id}};
}

void DeltaNetRecurrenceOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*attrs*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;

    CoreCoord core = {0, 0};

    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
        runtime_args[0] = tensor_args.state.buffer()->address();
        runtime_args[1] = tensor_args.conv_out.buffer()->address();
    }

    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
        runtime_args[0] = output_tensor.buffer()->address();
        runtime_args[1] = tensor_args.state.buffer()->address();
    }
}

}  // namespace ttnn::operations::experimental::deltanet
