// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::graph_kernel_op::detail {

namespace nb = nanobind;

void bind_experimental_graph_kernel_operation(nb::module_& mod);

}  // namespace ttnn::operations::experimental::graph_kernel_op::detail
