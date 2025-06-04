// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::operations::moreh::moreh_nll_loss_unreduced_backward {
namespace py = pybind11;
void bind_moreh_nll_loss_unreduced_backward_operation(py::module& module);

}  // namespace ttnn::operations::moreh::moreh_nll_loss_unreduced_backward
