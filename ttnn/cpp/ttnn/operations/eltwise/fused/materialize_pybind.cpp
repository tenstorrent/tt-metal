// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "materialize_pybind.hpp"
#include "ttnn/operations/eltwise/fused/device/materialize_device_operation.hpp"

#include "ttnn-pybind/decorators.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace ttnn::operations::fused {

void py_module(py::module& module) {
    auto prim = module.def_submodule("prim", "Primitive fused operations");
    auto materialize = ttnn::prim::materialize;
    using materialize_t = decltype(materialize);

    ttnn::decorators::bind_registered_operation(
        prim,
        materialize,
        "",
        ttnn::pybind_overload_t{
            [](materialize_t self, lazy::FunctionView expression) { return self(expression); },
            py::arg("expression"),
        });
}

}  // namespace ttnn::operations::fused
