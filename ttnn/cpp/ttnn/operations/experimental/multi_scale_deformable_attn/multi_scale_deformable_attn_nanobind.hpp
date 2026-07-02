// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace ttnn::operations::experimental::multi_scale_deformable_attn::detail {

void bind_multi_scale_deformable_attn(nb::module_& mod);

}  // namespace ttnn::operations::experimental::multi_scale_deformable_attn::detail
