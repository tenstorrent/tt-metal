// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/global_circular_buffer.hpp>

namespace ttnn::operations::dram_prefetcher {

struct DramPrefetcherParams {
    uint32_t num_layers = 0;
    bool enable_performance_mode = false;
    std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer> global_cb;
};

struct DramPrefetcherInputs {
    std::vector<Tensor> input_tensors;
};

}  // namespace ttnn::operations::dram_prefetcher
