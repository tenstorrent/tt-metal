// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric.hpp"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tt-metalium/tt_metal.hpp>

namespace ttnn::fabric {

void py_bind_fabric_api(pybind11::module& module) {
    py::enum_<tt::FabricConfig>(module, "FabricConfig")
        .value("DISABLED", tt::FabricConfig::DISABLED)
        .value("FABRIC_1D", tt::FabricConfig::FABRIC_1D)
        .value("FABRIC_2D", tt::FabricConfig::FABRIC_2D)
        .value("CUSTOM", tt::FabricConfig::CUSTOM);  // DISABLED = 0, FABRIC_1D = 1, FABRIC_2D = 2, CUSTOM = 4

    module.def("initialize_fabric_config", &tt::tt_metal::detail::InitializeFabricConfig, py::arg("config"));
}

}  // namespace ttnn::fabric
