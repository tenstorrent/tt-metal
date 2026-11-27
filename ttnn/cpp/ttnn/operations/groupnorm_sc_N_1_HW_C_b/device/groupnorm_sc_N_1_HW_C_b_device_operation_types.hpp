// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::groupnorm_sc_N_1_HW_C_b {

// What the caller chose and the entry resolved, as scalars. The solved blocking plan is not an attribute: it is
// memoized on its inputs (device/groupnorm_sc_N_1_HW_C_b_planner.hpp) and looked up by const&.
//
// The device operation writes its own compute_program_hash: eps is a per-call runtime scalar (compute per-core arg
// EPS_BITS), left out of the key and re-applied by the factory's override_runtime_arguments; every other attribute
// and the solved plan's structural fields are in the key.
struct GroupnormScN1HwCbParams {
    uint32_t num_groups = 1;
    float eps = 1e-5f;  // Python float; the kernel takes its fp32 bits
    bool in_place = false;
};

struct GroupnormScN1HwCbInputs {
    const Tensor& input;
    const std::optional<Tensor>& gamma;
    const std::optional<Tensor>& beta;
    // Preallocated output; must match the spec the op would allocate. May share the input's buffer (in place).
    const std::optional<Tensor>& output;
};

// Refusals of the support contract, named after their ttnn.operations._op_contract counterparts; the binding raises
// each as that Python type. Structural errors the Python raised as ValueError are std::invalid_argument.
struct SupportRefusal : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct UnsupportedAxisValue : SupportRefusal {
    using SupportRefusal::SupportRefusal;
};
// Reachable: SUPPORTED admits BLOCK_SHARDED and float32 / bfloat8_b / TILE affine / non-aligned shapes separately.
struct ExcludedCell : SupportRefusal {
    using SupportRefusal::SupportRefusal;
};
// The planner's NotImplementedError (no feasible column-slice count, a block-sharded layout it cannot run, L1 fit).
// The binding raises it as Python's builtin NotImplementedError.
struct PlanNotImplemented : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace ttnn::operations::groupnorm_sc_N_1_HW_C_b
