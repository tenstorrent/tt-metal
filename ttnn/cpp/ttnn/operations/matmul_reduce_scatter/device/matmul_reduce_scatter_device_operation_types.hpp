// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>

#include <tt-metalium/base_types.hpp>

#include "matmul_reduce_scatter_planner.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::matmul_reduce_scatter {

// Everything the public entry resolved before launch; all of it structural, so the default program key
// (every attribute plus the tensor specs) is exact.
struct MatmulReduceScatterParams {
    uint32_t cluster_axis = 0;
    int32_t scatter_dim = -2;  // canonical: -1 or -2
    // The topology as the program runs it (a 2-device ring is the line): Linear and a 2-device Ring share a program.
    ScheduleKind kind = ScheduleKind::LINE;
    uint32_t num_links = 0;  // L, resolved (None -> every usable link)
    // From the caller's ComputeConfigDescriptor (default HiFi2); fp32 DEST and full sync are always on.
    MathFidelity math_fidelity = MathFidelity::HiFi2;
    bool math_approx_mode = false;
    tt::tt_metal::MemoryConfig output_memory_config;  // default DRAM interleaved
    Plan plan;
};

struct MatmulReduceScatterInputs {
    const Tensor& input;
    const Tensor& weight;
    // The per-mesh persistent tensors (MeshResources): cb FINISHED / ARRIVALS_* back onto staging.
    const Tensor& staging;
    const Tensor& mailbox;
    // Preallocated output; must match the op's output spec exactly and may not share an input's buffer.
    const std::optional<Tensor>& output;
};

// Refusals, named after their Python counterparts. The binding raises UnsupportedAxisValue as
// ttnn.operations._op_contract.UnsupportedAxisValue and NotImplementedRefusal as NotImplementedError.
// (EXCLUSIONS is empty, so there is no ExcludedCell.) ValueErrors are std::invalid_argument.
struct UnsupportedAxisValue : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct NotImplementedRefusal : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace ttnn::operations::matmul_reduce_scatter
