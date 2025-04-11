// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_getitem_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_getitem/moreh_getitem.hpp"

namespace ttnn::operations::moreh::moreh_getitem {
void bind_moreh_getitem_operation(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::moreh_getitem,
        "Moreh Getitem operation",
        ttnn::nanobind_arguments_t{
            nb::arg("input"),
            nb::arg("index_tensors"),
            nb::arg("index_dims"),
            nb::kw_only(),
            nb::arg("output") = nb::none(),
            nb::arg("memory_config") = nb::none()});
}
}  // namespace ttnn::operations::moreh::moreh_getitem
