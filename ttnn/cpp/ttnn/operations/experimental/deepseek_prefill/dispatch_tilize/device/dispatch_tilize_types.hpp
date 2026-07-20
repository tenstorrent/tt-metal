// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_tilize {

struct DispatchTilizeParams {
    tt::tt_metal::DataType output_dtype;
    tt::tt_metal::MemoryConfig output_memory_config;
    uint32_t experts_per_chip;  // 0 => plain tilize (no region-aware skip)
};

struct DispatchTilizeInputs {
    ttnn::Tensor input_tensor;
    // Optional routing metadata for the region-aware (skip-padding) path. When both are
    // supplied the kernel bounds its work by the filled prefix of the dispatch buffer
    // (valid_rows = region_offsets[last] + align32(counts[last])); when omitted it tilizes
    // the full padded input (byte-identical to ttnn.to_layout).
    std::optional<ttnn::Tensor> expert_region_offsets;
    std::optional<ttnn::Tensor> total_counts_per_expert;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_tilize
