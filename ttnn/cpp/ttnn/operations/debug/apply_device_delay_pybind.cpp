// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "apply_device_delay_pybind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "apply_device_delay.hpp"
#include <tt-metalium/sub_device_types.hpp>

namespace ttnn::operations::debug {

void py_bind_apply_device_delay(py::module& module) {
    auto doc =
        R"doc(apply_device_delay(mesh_device: ttnn.MeshDevice, delays: List[List[int]], subdevice_id: Optional[ttnn.SubDeviceId] = None) -> None

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

    module.def(
        "apply_device_delay",
        [](MeshDevice& mesh_device,
           const std::vector<std::vector<uint32_t>>& delays,
           std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
            ttnn::operations::debug::apply_device_delay(mesh_device, delays, subdevice_id);
        },
        py::arg("mesh_device"),
        py::arg("delays"),
        py::kw_only(),
        py::arg("subdevice_id") = std::nullopt,
        doc);
}

}  // namespace ttnn::operations::debug
