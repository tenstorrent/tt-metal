// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/base_types.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "matmul_reduce_scatter_prefill_planner.hpp"

namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill {

// Everything the public entry resolved before launch. The plan is solved once by the entry (it also sizes the
// per-mesh staging tensor before launch) and is structural, so the default program hash covers it with the rest.
struct MatmulReduceScatterPrefillParams {
    uint32_t cluster_axis = 0;
    int32_t scatter_dim = -2;  // canonical (-1 or -2)
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear;
    uint32_t num_links = 1;  // resolved L
    detail::ScheduleKind kind = detail::ScheduleKind::Line;
    // the caller's compute config; fp32 DEST and full sync are always on
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi2;
    bool math_approx_mode = false;
    tt::tt_metal::MemoryConfig output_memory_config;
    detail::Plan plan;
};

struct MatmulReduceScatterPrefillInputs {
    const Tensor& input_tensor;
    const Tensor& weight;
    const std::optional<Tensor>& output_tensor;
};

}  // namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill
