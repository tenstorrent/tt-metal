// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/global_circular_buffer.hpp>

namespace ttnn::experimental {

// Decode-optimized matmul C = A @ B for L1 width-sharded operands (full, partial, or batched B layout).
// `global_cb`: optional DRAM-sender GlobalCircularBuffer supplying in1 from the tensor prefetcher
// (full width-sharded factory only; the weight must then be a DRAM ND-sharded tensor).
Tensor matmul_decode(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool partial_width_sharded = false,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb = std::nullopt);

}  // namespace ttnn::experimental
