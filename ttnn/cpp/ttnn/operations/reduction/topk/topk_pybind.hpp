// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::reduction::detail {
namespace py = pybind11;

void bind_reduction_topk_operation(py::module& module);

}  // namespace ttnn::operations::reduction::detail
