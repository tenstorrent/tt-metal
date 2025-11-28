// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "manual_seed.hpp"

#include "device/manual_seed_operation.hpp"

namespace ttnn::operations::reduction {

Tensor ExecuteManualSeed::invoke(
    const std::variant<uint32_t, Tensor>& seeds,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const std::optional<std::variant<uint32_t, Tensor>>& user_ids,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::prim::manual_seed(seeds, device, user_ids, sub_core_grids);
}

}  // namespace ttnn::operations::reduction
