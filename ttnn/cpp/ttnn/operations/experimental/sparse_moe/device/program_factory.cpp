// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Sparse MoE Expert program factory.
// Multi-core: 8 cores, each handles num_experts/8 experts.
// Each core reads input (shared), streams weight tiles per expert, writes output columns.

#include "sparse_moe_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::sparse_moe {

SparseMoeExpertOperation::SingleCore::cached_program_t SparseMoeExpertOperation::SingleCore::create(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& expert_gu = tensor_args.expert_gu;

    auto* input_buffer = input.buffer();
    auto* weights_buffer = expert_gu.buffer();
    auto* output_buffer = output_tensor.buffer();

    tt::tt_metal::Program program{};

    // Data formats
    tt::DataFormat input_df = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat weight_df = tt::tt_metal::datatype_to_dataformat_converter(expert_gu.dtype());
    tt::DataFormat output_df = input_df;  // output same as input (bf16)
    uint32_t input_tile_size = tt::tile_size(input_df);
    uint32_t weight_tile_size = tt::tile_size(weight_df);
    uint32_t output_tile_size = tt::tile_size(output_df);

    // Dimensions
    uint32_t num_experts = attrs.num_experts;                   // 64
    uint32_t hidden_dim = attrs.hidden_dim;                     // 2048
    uint32_t expert_inter = attrs.expert_inter_dim;             // 512
    uint32_t expert_width = 2 * expert_inter;                   // 1024 (gate + up)
    uint32_t k_tiles = hidden_dim / 32;                         // 64
    uint32_t tiles_per_expert = expert_width / 32;              // 32
    uint32_t total_col_tiles = num_experts * tiles_per_expert;  // 2048
    uint32_t out_sub = 8;                                       // output sub-block size (fits in DST)

    // Multi-core: use up to 64 cores (8×8 grid), 1 expert per core
    uint32_t max_cores = 64;
    uint32_t num_cores = std::min(num_experts, max_cores);
    while (num_experts % num_cores != 0 && num_cores > 1) {
        num_cores--;
    }
    uint32_t experts_per_core = num_experts / num_cores;

    // 2D core grid: up to 8 per row
    uint32_t cores_x = std::min(num_cores, (uint32_t)8);
    uint32_t cores_y = (num_cores + cores_x - 1) / cores_x;
    CoreRange core_range(CoreCoord(0, 0), CoreCoord(cores_x - 1, cores_y - 1));

    // ---- Circular Buffers ----
    // cb_input (c_0): input tiles, held for all experts. Size = k_tiles.
    CircularBufferConfig cb_input_config =
        CircularBufferConfig(k_tiles * input_tile_size, {{tt::CBIndex::c_0, input_df}})
            .set_page_size(tt::CBIndex::c_0, input_tile_size);
    CreateCircularBuffer(program, core_range, cb_input_config);

    // cb_weights (c_1): weight tiles, streamed out_sub at a time. Double-buffered.
    CircularBufferConfig cb_weights_config =
        CircularBufferConfig(2 * out_sub * weight_tile_size, {{tt::CBIndex::c_1, weight_df}})
            .set_page_size(tt::CBIndex::c_1, weight_tile_size);
    CreateCircularBuffer(program, core_range, cb_weights_config);

    // cb_out (c_16): output tiles, single-buffered.
    CircularBufferConfig cb_out_config = CircularBufferConfig(2 * output_tile_size, {{tt::CBIndex::c_16, output_df}})
                                             .set_page_size(tt::CBIndex::c_16, output_tile_size);
    CreateCircularBuffer(program, core_range, cb_out_config);

    // ---- Reader Kernel ----
    std::vector<uint32_t> reader_compile_args = {
        k_tiles,           // 0
        tiles_per_expert,  // 1
        out_sub,           // 2
        total_col_tiles,   // 3
    };
    TensorAccessorArgs(*input_buffer).append_to(reader_compile_args);    // 4+
    TensorAccessorArgs(*weights_buffer).append_to(reader_compile_args);  // next

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/sparse_moe/device/kernels/dataflow/reader_sparse_expert.cpp",
        core_range,
        ReaderDataMovementConfig(reader_compile_args));

    // ---- Writer Kernel ----
    std::vector<uint32_t> writer_compile_args = {};
    TensorAccessorArgs(*output_buffer).append_to(writer_compile_args);

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/sparse_moe/device/kernels/dataflow/writer_sparse_expert.cpp",
        core_range,
        WriterDataMovementConfig(writer_compile_args));

    // ---- Compute Kernel ----
    std::vector<uint32_t> compute_compile_args = {
        experts_per_core,  // 0
        k_tiles,           // 1
        tiles_per_expert,  // 2
        out_sub,           // 3
    };
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/sparse_moe/device/kernels/compute/sparse_expert_matmul.cpp",
        core_range,
        ComputeConfig{
            .math_fidelity = MathFidelity::LoFi, .math_approx_mode = true, .compile_args = compute_compile_args});

    // ---- Per-core Runtime Args ----
    for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++) {
        CoreCoord core(core_idx % cores_x, core_idx / cores_x);
        uint32_t expert_start = core_idx * experts_per_core;
        uint32_t tile_start = expert_start * tiles_per_expert;
        uint32_t num_output_tiles = experts_per_core * tiles_per_expert;

        // Reader: input_addr, weights_addr, expert_start, num_local_experts, active_flags[0..N-1]
        // For V1: all experts active (flags = 1)
        std::vector<uint32_t> reader_rt = {
            input_buffer->address(),
            weights_buffer->address(),
            expert_start,
            experts_per_core,
        };
        // Append per-expert active flags (all 1 for now — sparse skipping in V2)
        for (uint32_t e = 0; e < experts_per_core; e++) {
            reader_rt.push_back(1);  // active
        }
        SetRuntimeArgs(program, reader_kernel_id, core, reader_rt);

        // Writer: output_addr, tile_start, num_tiles
        SetRuntimeArgs(program, writer_kernel_id, core, {output_buffer->address(), tile_start, num_output_tiles});
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .compute_kernel_id = compute_kernel_id}};
}

void SparseMoeExpertOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;

    uint32_t num_cores = std::min(attrs.num_experts, (uint32_t)64);
    while (attrs.num_experts % num_cores != 0 && num_cores > 1) {
        num_cores--;
    }
    uint32_t cores_x = std::min(num_cores, (uint32_t)8);
    for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++) {
        CoreCoord core(core_idx % cores_x, core_idx / cores_x);

        {
            auto& rt = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
            rt[0] = tensor_args.input.buffer()->address();
            rt[1] = tensor_args.expert_gu.buffer()->address();
        }
        {
            auto& rt = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
            rt[0] = output_tensor.buffer()->address();
        }
    }
}

}  // namespace ttnn::operations::experimental::sparse_moe
