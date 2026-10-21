// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/quasar/common/thread_plan.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim::qsr {

// Attributes of the Quasar layer_norm. Mirrors ttnn::prim::LayerNormParams minus the RMSNorm /
// distributed / fused-activation knobs, which the Welford-only port does not carry.
struct LayerNormParams {
    float eps = 0.0f;
    tt::tt_metal::MemoryConfig output_mem_config;
    ttnn::prim::LayerNormProgramConfig program_config;  // shared with the original op
    DeviceComputeKernelConfig compute_kernel_config;
    std::optional<tt::tt_metal::DataType> dtype;
    ThreadPlan threads;
    // TEMPORARY (sim bring-up until Quasar has an llk_math_welfords_sfpu_entry.h): an extra compute include
    // path (a stub Welford entry header) and an extra compute define (LN_DEBUG_STATS_FROM_EPS makes the kernel
    // use mean = var = eps so the data-movement plan can be checked numerically). Attributes, not raw env
    // reads, so they are part of the program-cache key.
    std::string extra_include_path;
    std::string extra_define;
};

struct LayerNormInputs {
    Tensor input;
    std::optional<Tensor> residual_input_tensor;  // b
    std::optional<Tensor> weight;                 // gamma
    std::optional<Tensor> bias;                   // beta
    std::optional<Tensor> recip_tensor;           // reciprocal LUT for the Welford algorithm
};

}  // namespace ttnn::prim::qsr
