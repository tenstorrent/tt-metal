// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

ttnn::Tensor combine_fabric2d(
    ttnn::MeshDevice& device,
    uint32_t num_links = 2,
    uint32_t num_tokens = 100,
    uint32_t chunk_size_bytes = 14336,
    uint32_t num_slots = 32,
    uint32_t axis = 0,
    std::optional<tt::tt_fabric::Topology> topology = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d

namespace ttnn {
using operations::experimental::deepseek_prefill::combine_fabric2d::combine_fabric2d;
}  // namespace ttnn
