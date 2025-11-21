// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "manual_seed.hpp"

#include "device/manual_seed_operation.hpp"

namespace ttnn::operations::reduction {

Tensor ExecuteManualSeed::invoke(
    MeshDevice& device, std::variant<uint32_t, Tensor> seeds, std::optional<std::variant<uint32_t, Tensor>> user_ids) {
    return ttnn::prim::manual_seed(device, seeds, user_ids);
}

}  // namespace ttnn::operations::reduction
