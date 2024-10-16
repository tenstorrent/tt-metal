// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_getitem_pybind.hpp"

#include "pybind11/decorators.hpp"
#include "ttnn/operations/moreh/moreh_getitem/moreh_getitem.hpp"

namespace ttnn::operations::moreh::moreh_getitem {
void bind_moreh_getitem_operation(py::module& module) {
    bind_registered_operation(module,
                              ttnn::moreh_getitem,
                              "Moreh Getitem operation",
                              ttnn::pybind_arguments_t{py::arg("input"),
                                                       py::arg("index_tensors"),
                                                       py::arg("index_dims"),
                                                       py::kw_only(),
                                                       py::arg("output") = std::nullopt,
                                                       py::arg("memory_config") = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_getitem
