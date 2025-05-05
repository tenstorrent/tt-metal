// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fold_pybind.hpp"

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_fold/fold.hpp"

namespace ttnn::operations::moreh::moreh_fold {
void bind_moreh_fold_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_fold,
        "Moreh Fold Operation",
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::arg("output") = std::nullopt,
            py::arg("output_size"),
            py::arg("kernel_size"),
            py::arg("dilation") = std::vector<uint32_t>{1, 1},
            py::arg("padding") = std::vector<uint32_t>{0, 0},
            py::arg("stride") = std::vector<uint32_t>{1, 1},
            py::arg("memory_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_fold
