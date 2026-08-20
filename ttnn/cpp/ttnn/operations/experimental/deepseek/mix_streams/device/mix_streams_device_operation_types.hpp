// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::mix_streams {

struct MixStreamsParams {
    uint32_t num_streams;  // hc; the valid hc x hc region of the comb tile.
    MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct MixStreamsInputs {
    const Tensor& post;          // [B, S, hc, 1]
    const Tensor& comb;          // [B, S, hc, hc]
    const Tensor& sublayer_out;  // [B, S, 1, D]
    const Tensor& streams;       // [B, S, hc, D]
};

using MixStreamsTensorReturn = Tensor;  // [B, S, hc, D]

}  // namespace ttnn::operations::experimental::deepseek::mix_streams
