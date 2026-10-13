// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::experimental::kda::chain_affine_transforms::detail {

namespace nb = nanobind;
void bind_chain_affine_transforms(nb::module_&);

}  // namespace ttnn::operations::experimental::kda::chain_affine_transforms::detail
