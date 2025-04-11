// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <tt-metalium/fabric.hpp>

namespace ttnn::fabric {

void bind_fabric_api(nb::module_& mod) {
    nb::enum_<tt::tt_fabric::FabricConfig>(mod, "FabricConfig")
        .value("DISABLED", tt::tt_fabric::FabricConfig::DISABLED)
        .value("FABRIC_1D", tt::tt_fabric::FabricConfig::FABRIC_1D)
        .value("FABRIC_1D_RING", tt::tt_fabric::FabricConfig::FABRIC_1D_RING)
        .value("FABRIC_2D", tt::tt_fabric::FabricConfig::FABRIC_2D)
        .value("FABRIC_2D_DYNAMIC", tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC)
        .value("CUSTOM", tt::tt_fabric::FabricConfig::CUSTOM);  // DISABLED = 0, FABRIC_1D = 1, FABRIC_2D = 2, CUSTOM = 4
    nb::enum_<tt::tt_fabric::FabricReliabilityMode>(mod, "FabricReliabilityMode", R"(
        Specifies how the fabric initialization handles system health and configuration.
        Values:
            STRICT_INIT: Initialize fabric such that the live links/devices must exactly match the mesh graph descriptor. Any downed devices/links will result in errors.
            RELAXED_INIT: Initialize the fabric with flexibility towards downed links/devices. The fabric will initialize with fewer routing planes than in mesh graph descriptor, based on live links.
            DYNAMIC_RECONFIG: [Unsupported] Placeholder for dynamic (runtime) reconfiguration based on live system events (such as hardware failures).
        )")
        .value("STRICT_INIT", tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE)
        .value("RELAXED_INIT", tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE)
        .value("DYNAMIC_RECONFIG", tt::tt_fabric::FabricReliabilityMode::DYNAMIC_RECONFIGURATION_SETUP_MODE);

    nb::enum_<tt::tt_fabric::FabricTensixConfig>(mod, "FabricTensixConfig", R"(
        Specifies the fabric tensix configuration mode for mux functionality. Enabling a FabricTensixConfig will result in fabric permanently reserving additional worker cores for the duration of the workload.
        Values:
            DISABLED: Fabric tensix mux functionality is disabled (default).
            MUX: Enable fabric tensix mux mode for worker → mux → fabric router routing.
        )")
        .value("DISABLED", tt::tt_fabric::FabricTensixConfig::DISABLED)
        .value("MUX", tt::tt_fabric::FabricTensixConfig::MUX);

    mod.def(
        "set_fabric_config",
        &tt::tt_fabric::SetFabricConfig,
        nb::arg("config"),
        nb::arg("reliability_mode") = tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
        nb::arg("num_planes") = nb::none(),
        nb::arg("fabric_tensix_config") = tt::tt_fabric::FabricTensixConfig::DISABLED);
}

}  // namespace ttnn::fabric
