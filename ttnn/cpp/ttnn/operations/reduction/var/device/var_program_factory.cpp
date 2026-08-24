// Copyright (c) 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "var_program_factory.hpp"
#include "ttnn/operations/normalization/common/device/kernels/two_pass_stats.hpp"
#include <tt-metalium/program.hpp>

namespace ttnn::operations::reduction {

using namespace tt::tt_metal;

operation::ProgramWithCallbacks var_multi_core(
    const Tensor& input,
    Tensor& output,
    int dim,
    bool keepdim,
    bool population_variance) {

    uint32_t reduction_dim_size = input.logical_shape()[dim];
    bool use_two_pass = tt::tt_metal::stats::should_use_two_pass_stats(input, reduction_dim_size);

    if (use_two_pass) {
        return var_two_pass_program_factory(input, output, dim, keepdim, population_variance);
    }

    return var_welford_program_factory(input, output, dim, keepdim, population_variance);
}

operation::ProgramWithCallbacks var_two_pass_program_factory(
    const Tensor& input,
    Tensor& output,
    int dim,
    bool keepdim,
    bool population_variance) {
    
    Program program = CreateProgram();
    // Two-pass Variance implementation
    // Pass 1: Calculate shifted mean
    // Pass 2: Calculate variance using FP32 accumulation
    // L1-replay is enabled for the second pass to avoid DRAM re-reads
    
    uint32_t num_cores = input.device()->compute_with_storage_grid_size().x;
    
    std::vector<uint32_t> compile_time_args = {
        (uint32_t)input.buffer()->address(),
        (uint32_t)output.buffer()->address(),
        (uint32_t)population_variance
    };
    
    auto kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/var/device/kernels/var_two_pass.cpp",
        CoreRange(CoreCoord(0, 0), CoreCoord(num_cores - 1, 0)),
        CommonRuntimeArgs(compile_time_args));

    return {.program = std::move(program)};
}

operation::ProgramWithCallbacks var_welford_program_factory(
    const Tensor& input,
    Tensor& output,
    int dim,
    bool keepdim,
    bool population_variance) {
    Program program = CreateProgram();
    return {.program = std::move(program)};
}

} // namespace ttnn::operations::reduction