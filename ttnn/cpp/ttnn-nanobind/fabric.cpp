// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric.hpp"

#include <nanobind/nanobind.h>

#include <tt-metalium/tt_metal.hpp>

namespace ttnn::fabric {

void bind_fabric_api(nb::module_& mod) {
    nb::enum_<tt::tt_metal::FabricConfig>(mod, "FabricConfig")
        .value("DISABLED", tt::tt_metal::FabricConfig::DISABLED)
        .value("FABRIC_1D", tt::tt_metal::FabricConfig::FABRIC_1D)
        .value("FABRIC_1D_RING", tt::tt_metal::FabricConfig::FABRIC_1D_RING)
        .value("FABRIC_2D", tt::tt_metal::FabricConfig::FABRIC_2D)
        .value("CUSTOM", tt::tt_metal::FabricConfig::CUSTOM);  // DISABLED = 0, FABRIC_1D = 1, FABRIC_2D = 2, CUSTOM = 4

    mod.def(
        "set_fabric_config",
        &tt::tt_metal::detail::SetFabricConfig,
        nb::arg("config"),
        nb::arg("num_planes") = std::nullopt);
}

}  // namespace ttnn::fabric
