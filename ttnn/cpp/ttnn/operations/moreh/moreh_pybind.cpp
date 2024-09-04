// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_pybind.hpp"

#include "ttnn/operations/moreh/moreh_adam/moreh_adam_pybind.hpp"
#include "ttnn/operations/moreh/moreh_arange/moreh_arange_pybind.hpp"
#include "ttnn/operations/moreh/moreh_getitem/moreh_getitem_pybind.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum_pybind.hpp"
#include "ttnn/operations/moreh/moreh_mean/moreh_mean_pybind.hpp"
#include "ttnn/operations/moreh/moreh_mean_backward/moreh_mean_backward_pybind.hpp"
#include "ttnn/operations/moreh/moreh_dot_op/moreh_dot_pybind.hpp"

namespace ttnn::operations::moreh {
void bind_moreh_operations(py::module &module) {
    moreh_arange::bind_moreh_arange_operation(module);
    moreh_adam::bind_moreh_adam_operation(module);
    moreh_getitem::bind_moreh_getitem_operation(module);
    moreh_sum::bind_moreh_sum_operation(module);
    moreh_mean::bind_moreh_mean_operation(module);
    moreh_mean_backward::bind_moreh_mean_backward_operation(module);
    moreh_dot::bind_moreh_dot_operation(module);
}
}  // namespace ttnn::operations::moreh
