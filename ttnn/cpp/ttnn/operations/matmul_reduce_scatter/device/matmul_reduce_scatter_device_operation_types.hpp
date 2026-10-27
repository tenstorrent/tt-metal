// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>

#include <tt-metalium/base_types.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "matmul_reduce_scatter_planner.hpp"

namespace ttnn::operations::matmul_reduce_scatter {

// Everything the public entry resolved before launch. The entry solves the plan (it also sizes the per-mesh staging
// tensor before launch); the plan is structural, so the program hash covers it through its inputs.
struct MatmulReduceScatterParams {
    uint32_t cluster_axis = 0;
    int32_t scatter_dim = -2;  // canonical (-1 or -2)
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear;
    uint32_t num_links = 1;  // resolved L
    detail::ScheduleKind kind = detail::ScheduleKind::Line;
    // the caller's compute config; fp32 DEST and full sync are forced on, as the Python does
    MathFidelity math_fidelity = MathFidelity::HiFi2;
    bool math_approx_mode = false;
    tt::tt_metal::MemoryConfig output_memory_config;
    // The plan's inputs, not the plan: the solve is a pure function of them (memoized by solve_plan), so the default
    // program hash over these 14 scalars keys the program exactly, and a hot call neither copies nor hashes the plan's
    // per-chip schedules. plan() looks the solved plan up.
    detail::PlanInputs plan_inputs;

    const detail::Plan& plan() const { return detail::solve_plan(plan_inputs); }
};

struct MatmulReduceScatterInputs {
    const Tensor& input_tensor;
    const Tensor& weight;
    const std::optional<Tensor>& output_tensor;
};

// SUPPORTED refusals: raised in Python as ttnn.operations._op_contract.UnsupportedAxisValue (EXCLUSIONS is empty, so
// no ExcludedCell).
struct UnsupportedAxisValue : std::runtime_error {
    using std::runtime_error::runtime_error;
};
// The Python's NotImplementedError (sharded output memory_config, no plan fits L1): raised as NotImplementedError.
// ValueError cases throw std::invalid_argument.
struct NotImplemented : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace ttnn::operations::matmul_reduce_scatter
