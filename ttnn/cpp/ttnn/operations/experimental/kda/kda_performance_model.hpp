// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <span>
#include <vector>

#include "ttnn/api/ttnn/operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::kda_performance_model {

struct KdaFpuWork {
    double fpu_matrix_flops = 0.0;
    double fpu_multiply_ops = 0.0;
    double fpu_add_ops = 0.0;
    double fpu_reduction_ops = 0.0;
};

using KdaProfilerModel = tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>>;

KdaProfilerModel make_profiler_model(
    const KdaFpuWork& work,
    std::span<const Tensor* const> inputs,
    const std::vector<Tensor>& outputs,
    tt::tt_metal::MathFidelity math_fidelity);

}  // namespace ttnn::experimental::prim::kda_performance_model
