// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/ccl/ccl_pybind.hpp"

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/erisc_datamover_builder.hpp"

namespace ttnn::operations::ccl {

void py_bind_common(pybind11::module& module) {
    py::enum_<ttnn::ccl::Topology>(module, "Topology")
        .value("Ring", ttnn::ccl::Topology::Ring)
        .value("Linear", ttnn::ccl::Topology::Linear);
}

}  // namespace ttnn::operations::ccl
