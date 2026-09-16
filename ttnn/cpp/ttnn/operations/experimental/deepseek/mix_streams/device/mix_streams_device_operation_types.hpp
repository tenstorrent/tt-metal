// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::mix_streams {

struct MixStreamsParams {
    uint32_t num_streams;  // hc; the valid hc x hc region of the comb tile.
    MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
    // Untilize dest and emit ROW_MAJOR. Set when streams or sublayer_out is ROW_MAJOR
    // (decode: sublayer_out is RM while the residual streams are still TILE).
    bool untilize_out = false;

    static constexpr auto attribute_names =
        std::forward_as_tuple("num_streams", "output_mem_config", "compute_kernel_config", "untilize_out");
    auto attribute_values() const {
        return std::forward_as_tuple(num_streams, output_mem_config, compute_kernel_config, untilize_out);
    }
};

struct MixStreamsInputs {
    const Tensor& post;          // [B, S, hc, 1]
    const Tensor& comb;          // [B, S, hc, hc]
    const Tensor& sublayer_out;  // [B, S, 1, D]
    const Tensor& streams;       // [B, S, hc, D]
};

using MixStreamsTensorReturn = Tensor;  // [B, S, hc, D]

}  // namespace ttnn::operations::experimental::deepseek::mix_streams
