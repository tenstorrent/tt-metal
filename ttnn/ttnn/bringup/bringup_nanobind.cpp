// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup on the combine-overlap branch: the flat routed expert only.

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace ttnn::operations::bringup::detail {
void bind_flat_routed_expert(nb::module_& mod);
}
namespace ttnn::bringup {
void py_module(nb::module_& mod) { ::ttnn::operations::bringup::detail::bind_flat_routed_expert(mod); }
}  // namespace ttnn::bringup
