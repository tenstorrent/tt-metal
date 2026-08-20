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
// `global_cb_k_blocks`: how many GCB pages carry one receiver's weight slab. 1 (the default) is one
// page per slab, so the GCB must hold a whole slab. Higher values cut the slab into that many
// K-blocks and stream them, letting the GCB be smaller than a slab -- it must equal the
// `block_count` of the prefetch request that fills the GCB or the two sides deadlock.
Tensor matmul_decode(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool partial_width_sharded = false,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb = std::nullopt,
    uint32_t global_cb_k_blocks = 1);

}  // namespace ttnn::experimental
