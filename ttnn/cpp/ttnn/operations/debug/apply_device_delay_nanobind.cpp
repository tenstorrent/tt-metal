// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "apply_device_delay_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "apply_device_delay.hpp"
#include <tt-metalium/sub_device_types.hpp>

namespace ttnn::operations::debug {

void bind_apply_device_delay(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Applies per-device delays by launching a single-core kernel on each device in the mesh that spins
            for the specified number of cycles. The shape of `delays` must match the mesh view shape
            (rows x cols), e.g. for an 8x4 mesh, delays.size()==8 and delays[r].size()==4.
            If `subdevice_id` is provided, the kernel will be scheduled on a worker core belonging to that subdevice.
            This will only guarantee a minimum skew, not a specific skew due to variation and host-side overhead.

            Args:
                mesh_device (ttnn.MeshDevice): The mesh device to apply delays to.
                delays (List[List[int]]): A 2D list of delay cycles, where delays[row][col] specifies the delay for device at position (row, col) in the mesh.
                subdevice_id (ttnn.SubDeviceId, optional): The subdevice ID for the subdevice on which we schedule the worker core. Defaults to `None`.

            Returns:
                None: This function does not return a value.

            Example:
                >>> # For a 2x2 mesh, apply different delays to each device
                >>> delays = [[1000, 2000], [3000, 4000]]  # cycles
                >>> ttnn.apply_device_delay(mesh_device, delays)
        )doc";

    mod.def(
        "apply_device_delay",
        [](MeshDevice& mesh_device,
           const std::vector<std::vector<uint32_t>>& delays,
           std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
            ttnn::operations::debug::apply_device_delay(mesh_device, delays, subdevice_id);
        },
        nb::arg("mesh_device"),
        nb::arg("delays"),
        nb::kw_only(),
        nb::arg("subdevice_id") = nb::none(),
        doc);
}

}  // namespace ttnn::operations::debug
