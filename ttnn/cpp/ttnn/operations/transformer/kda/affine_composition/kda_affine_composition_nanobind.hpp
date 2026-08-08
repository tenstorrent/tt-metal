// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::transformer {

namespace nb = nanobind;
void bind_kda_affine_composition(nb::module_&);

}  // namespace ttnn::operations::transformer
