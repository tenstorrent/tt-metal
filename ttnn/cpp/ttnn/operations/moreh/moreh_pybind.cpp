// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_pybind.hpp"

#include "ttnn/operations/moreh/moreh_cumsum/moreh_cumsum_pybind.hpp"

namespace ttnn::operations::moreh {
void bind_moreh_operations(py::module &module) {
    moreh_cumsum::bind_moreh_cumsum_operation(module);
    moreh_cumsum::bind_moreh_cumsum_backward_operation(module);
}
}  // namespace ttnn::operations::moreh::moreh_cumsum
