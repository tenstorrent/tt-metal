// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::prim {

struct SdpaWindowedParams {
    std::optional<float> scale;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct SdpaWindowedInputs {
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor cu_window_seqlens;

    static constexpr auto attribute_names = std::forward_as_tuple("q", "k", "v", "cu_window_seqlens");
    auto attribute_values() const { return std::forward_as_tuple(q, k, v, cu_window_seqlens); }
};

}  // namespace ttnn::prim
