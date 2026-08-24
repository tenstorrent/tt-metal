// Copyright (c) 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_program_factory.hpp"
#include "ttnn/operations/normalization/common/device/kernels/two_pass_stats.hpp"
#include <tt-metalium/program.hpp>

namespace ttnn::operations::normalization {

using namespace tt::tt_metal;

operation::ProgramWithCallbacks groupnorm_multi_core(
    const Tensor& input,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    Tensor& output,
    uint32_t num_groups,
    float eps,
    CoreCoord compute_with_storage_grid_size) {

    uint32_t group_size = input.logical_shape()[-1] / num_groups;
    bool use_two_pass = tt::tt_metal::stats::should_use_two_pass_stats(input, group_size);

    if (use_two_pass) {
        return groupnorm_two_pass_program_factory(input, gamma, beta, output, num_groups, eps, compute_with_storage_grid_size);
    }

    return groupnorm_welford_program_factory(input, gamma, beta, output, num_groups, eps, compute_with_storage_grid_size);
}

operation::ProgramWithCallbacks groupnorm_two_pass_program_factory(
    const Tensor& input,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    Tensor& output,
    uint32_t num_groups,
    float eps,
    CoreCoord compute_with_storage_grid_size) {
    
    Program program = CreateProgram();
    // Two-pass GroupNorm implementation
    // Pass 1: Shifted FP32 accumulation for mean per group
    // Pass 2: Variance calculation and normalization per group
    
    auto shard_spec = input.shard_spec().value();
    uint32_t num_cores = shard_spec.grid.num_cores();
    
    std::vector<uint32_t> compile_time_args = {
        (uint32_t)input.buffer()->address(),
        (uint32_t)output.buffer()->address(),
        num_groups,
        (uint32_t)(eps * 1000000.0f)
    };
    
    auto kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/groupnorm_two_pass.cpp",
        CoreRange(CoreCoord(0, 0), CoreCoord(num_cores - 1, 0)),
        CommonRuntimeArgs(compile_time_args));

    return {.program = std::move(program)};
}

operation::ProgramWithCallbacks groupnorm_welford_program_factory(
    const Tensor& input,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    Tensor& output,
    uint32_t num_groups,
    float eps,
    CoreCoord compute_with_storage_grid_size) {
    Program program = CreateProgram();
    return {.program = std::move(program)};
}

} // namespace ttnn::operations::normalization