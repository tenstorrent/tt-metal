// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <tt-metalium/host_api.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "layernorm_types.hpp"

namespace ttnn::prim {

std::pair<std::optional<tt::tt_metal::Tensor>, uint32_t> create_reciprocal_tensor_if_needed(
    tt::tt_metal::IDevice* device, uint32_t W, const tt::tt_metal::CoreRangeSet& cores, bool use_welford);

// Creates a program config from shard spec.
// - If shard_spec has value, creates a sharded config derived from it
// - Otherwise, returns a default interleaved config
LayerNormProgramConfig create_layernorm_program_config(const std::optional<tt::tt_metal::ShardSpec>& shard_spec);

}  // namespace ttnn::prim
