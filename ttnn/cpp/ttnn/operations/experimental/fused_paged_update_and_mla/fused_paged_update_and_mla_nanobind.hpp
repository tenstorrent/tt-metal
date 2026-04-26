// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::fused_paged_update_and_mla {
void bind_fused_paged_update_and_mla(nb::module_& mod);
} // namespace ttnn::operations::experimental::fused_paged_update_and_mla
