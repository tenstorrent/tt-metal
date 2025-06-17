// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric.hpp"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tt-metalium/tt_metal.hpp>

namespace ttnn::fabric {

void py_bind_fabric_api(py::module& module) {
    py::enum_<tt::tt_metal::FabricConfig>(module, "FabricConfig")
        .value("DISABLED", tt::tt_metal::FabricConfig::DISABLED)
        .value("FABRIC_1D", tt::tt_metal::FabricConfig::FABRIC_1D)
        .value("FABRIC_1D_RING", tt::tt_metal::FabricConfig::FABRIC_1D_RING)
        .value("FABRIC_2D", tt::tt_metal::FabricConfig::FABRIC_2D)
        .value("CUSTOM", tt::tt_metal::FabricConfig::CUSTOM);  // DISABLED = 0, FABRIC_1D = 1, FABRIC_2D = 2, CUSTOM = 4

    module.def(
        "set_fabric_config",
        &tt::tt_metal::detail::SetFabricConfig,
        py::arg("config"),
        py::arg("num_planes") = std::nullopt);
}

}  // namespace ttnn::fabric
