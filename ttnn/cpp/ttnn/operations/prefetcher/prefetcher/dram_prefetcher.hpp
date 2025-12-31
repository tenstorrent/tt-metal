// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/decorators.hpp"

#include <tt-metalium/global_circular_buffer.hpp>

namespace ttnn::operations::dram_prefetcher {

ttnn::Tensor dram_prefetcher(
    std::vector<ttnn::Tensor>& tensors,
    uint32_t num_layers,
    const std::optional<const GlobalCircularBuffer>& global_cb,
    bool enable_performance_mode = false);

}  // namespace ttnn::operations::dram_prefetcher

namespace ttnn {

using ttnn::operations::dram_prefetcher::dram_prefetcher;

}  // namespace ttnn
