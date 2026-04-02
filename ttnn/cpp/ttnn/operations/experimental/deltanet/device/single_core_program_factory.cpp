// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Multi-core DeltaNet recurrence: distributes heads across cores.
// Each core handles (num_heads / num_cores) heads independently.
// This replaces the original single-core implementation for ~8x speedup.

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

    // Multi-core: distribute heads across cores. Use env var to tune.
    uint32_t max_cores = 24;  // 24 cores = 2 heads/core is sweet spot for 48 heads
    const char* env_cores = std::getenv("DELTANET_CORES");
    if (env_cores) {
        max_cores = std::atoi(env_cores);
    }
    uint32_t num_cores = std::min(num_heads, max_cores);
    while (num_heads % num_cores != 0 && num_cores > 1) {
        num_cores--;
    }
    uint32_t heads_per_core = num_heads / num_cores;

    // Layout cores in 2D grid: up to 8 per row (WH has 8 cols)
    uint32_t cores_x = std::min(num_cores, (uint32_t)8);
    uint32_t cores_y = (num_cores + cores_x - 1) / cores_x;
    CoreRange core_range(CoreCoord(0, 0), CoreCoord(cores_x - 1, cores_y - 1));

    // ---- Circular Buffers (same config per core, double-buffered = 2 tiles) ----
    // IMPORTANT: Larger CBs (>2 tiles for I/O) cause device hangs. Keep double-buffered.
    CircularBufferConfig cb_q_config = CircularBufferConfig(2 * tile_size, {{tt::CBIndex::c_0, cb_data_format}})
                                           .set_page_size(tt::CBIndex::c_0, tile_size);
    CreateCircularBuffer(program, core_range, cb_q_config);
    CircularBufferConfig cb_k_config = CircularBufferConfig(2 * tile_size, {{tt::CBIndex::c_1, cb_data_format}})
                                           .set_page_size(tt::CBIndex::c_1, tile_size);
    CreateCircularBuffer(program, core_range, cb_k_config);
    CircularBufferConfig cb_v_config = CircularBufferConfig(2 * tile_size, {{tt::CBIndex::c_2, cb_data_format}})
                                           .set_page_size(tt::CBIndex::c_2, tile_size);
    CreateCircularBuffer(program, core_range, cb_v_config);
    CircularBufferConfig cb_decay_config = CircularBufferConfig(2 * tile_size, {{tt::CBIndex::c_3, cb_data_format}})
                                               .set_page_size(tt::CBIndex::c_3, tile_size);
    CreateCircularBuffer(program, core_range, cb_decay_config);
    CircularBufferConfig cb_beta_config = CircularBufferConfig(2 * tile_size, {{tt::CBIndex::c_4, cb_data_format}})
                                              .set_page_size(tt::CBIndex::c_4, tile_size);
    CreateCircularBuffer(program, core_range, cb_beta_config);
    CircularBufferConfig cb_state_config = CircularBufferConfig(2 * tile_size, {{tt::CBIndex::c_5, cb_data_format}})
                                               .set_page_size(tt::CBIndex::c_5, tile_size);
    CreateCircularBuffer(program, core_range, cb_state_config);
    CircularBufferConfig cb_out_config = CircularBufferConfig(2 * tile_size, {{tt::CBIndex::c_16, cb_data_format}})
                                             .set_page_size(tt::CBIndex::c_16, tile_size);
    CreateCircularBuffer(program, core_range, cb_out_config);
    CircularBufferConfig cb_state_out_config =
        CircularBufferConfig(2 * tile_size, {{tt::CBIndex::c_17, cb_data_format}})
            .set_page_size(tt::CBIndex::c_17, tile_size);
    CreateCircularBuffer(program, core_range, cb_state_out_config);

    // cb_scratch (c_24): working state for matmul recurrence (compute-only, per head)
    CircularBufferConfig cb_scratch_config =
        CircularBufferConfig(state_tiles_per_head * tile_size, {{tt::CBIndex::c_24, cb_data_format}})
            .set_page_size(tt::CBIndex::c_24, tile_size);
    CreateCircularBuffer(program, core_range, cb_scratch_config);

    // cb_scratch2 (c_25): Q tile cache for matmul (compute-only, k_tiles capacity)
    CircularBufferConfig cb_scratch2_config =
        CircularBufferConfig(k_tiles * tile_size, {{tt::CBIndex::c_25, cb_data_format}})
            .set_page_size(tt::CBIndex::c_25, tile_size);
    CreateCircularBuffer(program, core_range, cb_scratch2_config);

    // ---- Reader Kernel ----
    // Compile-time: heads_per_core (not total num_heads), k_tiles, v_tiles, TensorAccessorArgs
    std::vector<uint32_t> reader_compile_args = {
        heads_per_core,
        k_tiles,
        v_tiles,
    };
    TensorAccessorArgs(*state_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(*conv_out_buffer).append_to(reader_compile_args);

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/dataflow/reader_multicore.cpp",
        core_range,
        ReaderDataMovementConfig(reader_compile_args));

    // ---- Writer Kernel ----
    std::vector<uint32_t> writer_compile_args = {
        heads_per_core,
        v_tiles,
        state_tiles_per_head,
    };
    TensorAccessorArgs(*output_buffer).append_to(writer_compile_args);
    TensorAccessorArgs(*state_buffer).append_to(writer_compile_args);

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/dataflow/writer_multicore.cpp",
        core_range,
        WriterDataMovementConfig(writer_compile_args));

    // ---- Compute Kernel ----
    // Each core processes heads_per_core heads (same compute kernel, fewer iterations)
    std::vector<uint32_t> compute_compile_args = {
        heads_per_core,
        k_tiles,
        v_tiles,
    };
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deltanet/device/kernels/compute/deltanet_recurrence.cpp",
        core_range,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2, .math_approx_mode = true, .compile_args = compute_compile_args});

    // ---- Per-core Runtime Args (head_start differs per core) ----
    for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++) {
        CoreCoord core(core_idx % cores_x, core_idx / cores_x);
        uint32_t head_start = core_idx * heads_per_core;

        SetRuntimeArgs(
            program, reader_kernel_id, core, {state_buffer->address(), conv_out_buffer->address(), head_start});

        SetRuntimeArgs(
            program, writer_kernel_id, core, {output_buffer->address(), state_buffer->address(), head_start});
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .compute_kernel_id = compute_kernel_id}};
}

void DeltaNetRecurrenceOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;

    // Recompute core count (must match create())
    uint32_t max_cores = 24;
    const char* env_cores = std::getenv("DELTANET_CORES");
    if (env_cores) {
        max_cores = std::atoi(env_cores);
    }
    uint32_t num_cores = std::min(attrs.num_heads, max_cores);
    while (attrs.num_heads % num_cores != 0 && num_cores > 1) {
        num_cores--;
    }
    uint32_t heads_per_core = attrs.num_heads / num_cores;
    uint32_t cores_x = std::min(num_cores, (uint32_t)8);

    for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++) {
        CoreCoord core(core_idx % cores_x, core_idx / cores_x);
        uint32_t head_start = core_idx * heads_per_core;

        {
            auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = tensor_args.state.buffer()->address();
            runtime_args[1] = tensor_args.conv_out.buffer()->address();
            runtime_args[2] = head_start;
        }
        {
            auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = output_tensor.buffer()->address();
            runtime_args[1] = tensor_args.state.buffer()->address();
            runtime_args[2] = head_start;
        }
    }
}

}  // namespace ttnn::operations::experimental::deltanet
