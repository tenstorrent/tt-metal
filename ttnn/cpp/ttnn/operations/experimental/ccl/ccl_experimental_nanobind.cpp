// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ccl_experimental_nanobind.hpp"

#include <nanobind/nanobind.h>

namespace ttnn::operations::experimental::ccl {

// The experimental CCL ops were removed for agent evaluation (TODO(nuked-op ccl)); the
// `ttnn.experimental` CCL submodule is kept, empty, so the module layout stays the same.
void py_module(nb::module_& /*mod*/) {}

}  // namespace ttnn::operations::experimental::ccl
