// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::all_gather_rms_norm {

// What the caller chose and the entry resolved, as scalars. Structural fields are in the program key; epsilon is a
// per-call scalar (worker common arg EPS_BITS): the custom compute_program_hash leaves it out and
// override_runtime_arguments re-applies it.
struct AllGatherRmsNormParams {
    uint32_t cluster_axis = 0;
    bool is_layer_norm = false;
    double epsilon = 1e-6;  // Python float; packed to fp32 bits for the kernels
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear;
    uint32_t num_links = 1;  // resolved: min(usable links, 2) when the caller passed None
    // compute config the Python reads (math_fidelity, fp32_dest_acc_en, math_approx_mode); default HiFi4 / true / false
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
    bool fp32_dest_acc_en = true;
    bool math_approx_mode = false;
    // output placement when no output_tensor is given (Python default: DRAM interleaved)
    tt::tt_metal::MemoryConfig output_memory_config;
};

struct AllGatherRmsNormInputs {
    const Tensor& input;
    const std::optional<Tensor>& gamma;
    const std::optional<Tensor>& beta;
    const std::optional<Tensor>& output;
    // R_U (W_local % 32 != 0) staging: DRAM [1, 1, G * R * 32, Wlt * 32] of the output dtype; caller-held or
    // allocated by the entry for the call.
    const std::optional<Tensor>& scratch;
};

// Support refusals, named after ttnn.operations._op_contract. Both are reachable: UnsupportedAxisValue for an axis
// value outside SUPPORTED (and the placement / L1-fit refusals), ExcludedCell for (float32, fp32_dest_acc_en=False)
// and (bfloat8_b, w_non_aligned). Caller errors the Python raised as ValueError throw std::invalid_argument.
struct SupportRefusal : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct UnsupportedAxisValue : SupportRefusal {
    using SupportRefusal::SupportRefusal;
};
struct ExcludedCell : SupportRefusal {
    using SupportRefusal::SupportRefusal;
};

}  // namespace ttnn::operations::all_gather_rms_norm
