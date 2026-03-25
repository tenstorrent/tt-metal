// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "forward_substitution.hpp"

#include "device/forward_substitution_device_operation.hpp"

namespace ttnn {

Tensor forward_substitution(const Tensor& input, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::forward_substitution(input, memory_config);
}

}  // namespace ttnn
