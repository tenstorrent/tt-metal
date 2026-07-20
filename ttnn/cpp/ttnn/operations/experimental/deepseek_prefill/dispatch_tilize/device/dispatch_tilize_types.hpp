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
    uint32_t experts_per_chip;  // groups the [1,E] counts into per-chip fills; must be >0 on the skip path
};

struct DispatchTilizeInputs {
    ttnn::Tensor input_tensor;
    // Optional routing metadata for the region-aware (skip-padding) path. When supplied the kernel bounds its
    // work by the filled prefix of the dispatch buffer: valid_rows = max_chip Σ_{e∈chip} align32(count[e]),
    // the fullest chip's fill (chips = consecutive experts_per_chip groups). Omitted => full tilize
    // (byte-identical to ttnn.to_layout).
    std::optional<ttnn::Tensor> total_counts_per_expert;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_tilize
