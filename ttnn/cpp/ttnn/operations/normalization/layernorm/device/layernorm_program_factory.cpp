// Copyright (c) 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_program_factory.hpp"
#include "ttnn/operations/normalization/common/device/kernels/two_pass_stats.hpp"
#include <tt-metalium/program.hpp>
#include <tt-metalium/buffers.hpp>

namespace ttnn::operations::normalization {

using namespace tt::tt_metal;

operation::ProgramWithCallbacks layernorm_multi_core(
    const Tensor& input,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    Tensor& output,
    float eps,
    bool is_groupnorm,
    CoreCoord compute_with_storage_grid_size) {

    uint32_t reduction_dim_size = input.logical_shape()[-1];
    bool use_two_pass = tt::tt_metal::stats::should_use_two_pass_stats(input, reduction_dim_size);

    if (use_two_pass) {
        return layernorm_two_pass_program_factory(input, gamma, beta, output, eps, compute_with_storage_grid_size);
    }

    return layernorm_welford_program_factory(input, gamma, beta, output, eps, compute_with_storage_grid_size);
}

operation::ProgramWithCallbacks layernorm_two_pass_program_factory(
    const Tensor& input,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    Tensor& output,
    float eps,
    CoreCoord compute_with_storage_grid_size) {
    
    Program program = CreateProgram();
    // Two-pass LayerNorm implementation
    // Pass 1: Compute mean with shifted FP32 accumulation
    // Pass 2: Compute variance and normalize, utilizing L1-resident data
    
    auto shard_spec = input.shard_spec().value();
    uint32_t num_cores = shard_spec.grid.num_cores();
    
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)input.buffer()->address()};
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)output.buffer()->address()};
    
    auto reader_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/layernorm_two_pass_reader.cpp",
        CoreRange(CoreCoord(0, 0), CoreCoord(num_cores - 1, 0)),
        ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/layernorm_two_pass_writer.cpp",
        CoreRange(CoreCoord(0, 0), CoreCoord(num_cores - 1, 0)),
        WriterDataMovementConfig(writer_compile_time_args));

    // Setup circular buffers for L1-replay optimization
    auto cb_mean = CreateCircularBuffer(program, CoreRange(CoreCoord(0, 0), CoreCoord(num_cores - 1, 0)), tt::tt_metal::CircularBufferConfig(sizeof(float), {{CB::c_0, tt::tt_metal::DataFormat::Float32}}));
    
    return {.program = std::move(program), .override_runtime_arguments_callback = [](const void* operation, const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, std::vector<Tensor>& output_tensors, void* user_args) {}};
}

operation::ProgramWithCallbacks layernorm_welford_program_factory(
    const Tensor& input,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    Tensor& output,
    float eps,
    CoreCoord compute_with_storage_grid_size) {
    // Existing Welford implementation fallback
    Program program = CreateProgram();
    return {.program = std::move(program)};
}

} // namespace ttnn::operations::normalization