// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "simple_add.hpp"

#include "device/simple_add_device_operation.hpp"

namespace ttnn::operations::experimental::quasar {

Tensor simple_add(const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::qsr::simple_add(input_a, input_b, memory_config.value_or(input_a.memory_config()));
}

}  // namespace ttnn::operations::experimental::quasar
