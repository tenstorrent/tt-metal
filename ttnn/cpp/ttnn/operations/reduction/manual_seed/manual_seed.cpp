// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "manual_seed.hpp"

namespace ttnn::operations::reduction {

uint32_t ExecuteManualSeed::invoke(
    MeshDevice& device, std::variant<uint32_t, Tensor> seeds, std::optional<std::variant<uint32_t, Tensor>> user_ids) {
    return 0;
    // TODO: Implementation
}

}  // namespace ttnn::operations::reduction
