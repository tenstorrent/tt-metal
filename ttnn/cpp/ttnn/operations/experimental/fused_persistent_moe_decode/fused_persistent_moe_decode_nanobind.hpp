// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::fused_persistent_moe_decode {
void bind_fused_persistent_moe_decode(nb::module_& mod);
} // namespace ttnn::operations::experimental::fused_persistent_moe_decode
