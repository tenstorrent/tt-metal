// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

// x [.., R, 2 hd] float32 TILE DRAM interleaved (R % 64 == 0, hd % 32 == 0; columns [kv | gate]) ->
// [.., R / 2, hd] float32 TILE: out[t] = kv[2t] w0 + kv[2t+1] w1, (w0, w1) = softmax(gate[2t], gate[2t+1]).
// output_tensor: the caller's buffer of exactly that spec, returned as is. memory_config: default DRAM.
// max_grid: cut of the compute grid (default the whole grid). Refusals throw PairPoolAssertion.
Tensor pair_pool(
    const Tensor& x,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<CoreCoord>& max_grid = std::nullopt);

}  // namespace ttnn
