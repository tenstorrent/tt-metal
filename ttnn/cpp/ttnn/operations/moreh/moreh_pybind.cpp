// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_pybind.hpp"

#include "ttnn/operations/moreh/moreh_adam/moreh_adam_pybind.hpp"

namespace ttnn::operations::moreh {
void bind_moreh_operations(py::module &module) { moreh_adam::bind_moreh_adam_operation(module); }
}  // namespace ttnn::operations::moreh
