// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn-nanobind/export_enum.hpp"

namespace ttnn::fabric {

void bind_fabric_api(nb::module_& mod) {
    export_enum<tt::tt_fabric::FabricConfig>(mod, "FabricConfig");

    export_enum<tt::tt_fabric::FabricReliabilityMode>(mod, "FabricReliabilityMode", R"(
        Specifies how the fabric initialization handles system health and configuration.
        Values:
            STRICT_INIT: Initialize fabric such that the live links/devices must exactly match the mesh graph descriptor. Any downed devices/links will result in errors.
            RELAXED_INIT: Initialize the fabric with flexibility towards downed links/devices. The fabric will initialize with fewer routing planes than in mesh graph descriptor, based on live links.
            DYNAMIC_RECONFIG: [Unsupported] Placeholder for dynamic (runtime) reconfiguration based on live system events (such as hardware failures).
        )");

    export_enum<tt::tt_fabric::FabricTensixConfig>(mod, "FabricTensixConfig", R"(
        Specifies the fabric tensix configuration mode for mux functionality. Enabling a FabricTensixConfig will result in fabric permanently reserving additional worker cores for the duration of the workload.
        Values:
            DISABLED: Fabric tensix mux functionality is disabled (default).
            MUX: Enable fabric tensix mux mode for worker → mux → fabric router routing.
        )");

    export_enum<tt::tt_fabric::FabricUDMMode>(mod, "FabricUDMMode", R"(
        Specifies the Unified Datamovement mode for configuring fabric with different parameters.
        Values:
            DISABLED: UDM mode is disabled (default).
            ENABLED: UDM mode is enabled.
        )");

    mod.def(
        "set_fabric_config",
        &tt::tt_fabric::SetFabricConfig,
        nb::arg("config"),
        nb::arg("reliability_mode") = nb::cast(tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE),
        nb::arg("num_planes") = nb::none(),
        nb::arg("fabric_tensix_config") = nb::cast(tt::tt_fabric::FabricTensixConfig::DISABLED),
        nb::arg("fabric_udm_mode") = nb::cast(tt::tt_fabric::FabricUDMMode::DISABLED));
}

}  // namespace ttnn::fabric
