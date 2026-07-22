// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reallocate.hpp"

#include "ttnn/operations/experimental/quasar/move/move.hpp"

namespace ttnn::operations::experimental::quasar {

Tensor reallocate(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::operations::experimental::quasar::move(input_tensor, memory_config);
}

}  // namespace ttnn::operations::experimental::quasar
