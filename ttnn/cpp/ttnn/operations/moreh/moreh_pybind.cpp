// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/moreh/moreh_matmul/moreh_matmul_pybind.hpp"
#include "ttnn/operations/moreh/moreh_bmm/moreh_bmm_pybind.hpp"
#include "ttnn/operations/moreh/moreh_bmm_backward/moreh_bmm_backward_pybind.hpp"

namespace ttnn::operations::moreh {
void py_module(py::module& module) {
    moreh_matmul::bind_moreh_matmul_operation(module);
    moreh_bmm::bind_moreh_bmm_operation(module);
    moreh_bmm_backward::bind_moreh_bmm_backward_operation(module);
}
}  // namespace ttnn::operations::moreh
