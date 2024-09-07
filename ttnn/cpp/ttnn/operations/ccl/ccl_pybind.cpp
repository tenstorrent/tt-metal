// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/ccl/ccl_pybind.hpp"
#include "ttnn/operations/ccl/ccl_fabric.hpp"
#include "ttnn/cpp/pybind11/export_enum.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather_pybind.hpp"
#include "ttnn/operations/ccl/line_all_gather/line_all_gather_pybind.hpp"
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>


namespace ttnn {
namespace operations {
namespace ccl {

void py_module(py::module& module) {
    py::enum_<ttnn::ccl::OpFabricMode>(module, "FabricMode")
        .value("EDM", ttnn::ccl::OpFabricMode::TEMPORARY_EDM)
        .value("PersistentEDM", ttnn::ccl::OpFabricMode::PERSISTENT_EDM);

    ccl::py_bind_all_gather(module);
    ccl::py_bind_line_all_gather(module);
    ccl::py_bind_reduce_scatter(module);

}

}  // namespace ccl
}  // namespace operations
}  // namespace ttn
