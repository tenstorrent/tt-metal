// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>  // for tt::tt_fabric::Topology
#include "all_gather.hpp"

namespace ttnn::operations::ccl {

// Explicit cast to resolve ambiguous overload with deprecated function.
// Delete this once the deprecated function is deleted.
using AllGatherFn = ttnn::Tensor (*)(
    const ttnn::Tensor&,
    int32_t,
    std::optional<uint32_t>,
    const std::optional<ttnn::MemoryConfig>&,
    const std::optional<ttnn::Tensor>&,
    const std::optional<tt::tt_metal::SubDeviceId>&,
    const std::optional<CoreRangeSet>&,
    std::optional<uint32_t>,
    std::optional<tt::tt_fabric::Topology>,
    std::optional<uint32_t>,
    std::optional<uint32_t>,
    std::optional<uint32_t>,
    bool);

void bind_all_gather(nb::module_& mod) {
    const auto* doc = R"doc(
        Performs an all-gather collective operation that gathers data from all devices into a new output tensor, concatenated along the specified :attr:`dim`. If the :attr:`input_tensor` has unaligned row-major pages or padded tiles on the gather :attr:`dim`, a slower composite all-gather implementation is used.

        Args:
            input_tensor (ttnn.Tensor): Input tensor to be gathered.
            dim (int): Dimension along which to concatenate.

        Keyword Args:
            cluster_axis (int, optional): Axis on the 2D mesh device grid to gather along. Each of the non-cluster_axis dimensions perform independent all-gathers along the devices on the cluster_axis. Irrelevant for 1D mesh grids.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output tensor. Defaults to the input tensor's memory config.
            output_tensor (ttnn.Tensor, optional): Pre-allocated output tensor, can improve performance if provided. This must be allocated before invoking any op to avoid races. Defaults to None (op allocates a new output tensor).
            subdevice_id (ttnn.SubDeviceId, optional): Subdevice id for worker cores. Defaults to the first subdevice on the mesh device.
            sub_core_grids (CoreRangeSet, optional): Restricts worker core selection to this sub-grid. Defaults to all cores on the chosen subdevice.
            num_links (int, optional): Deprecated and ignored; retained for backward compatibility. Will be removed in a future release.
            topology (ttnn.Topology, optional): Deprecated and ignored; retained for backward compatibility. Will be removed in a future release.
            chunks_per_sync (int, optional): Deprecated and ignored; retained for backward compatibility. Will be removed in a future release.
            num_workers_per_link (int, optional): Deprecated and ignored; retained for backward compatibility. Will be removed in a future release.
            num_buffers_per_channel (int, optional): Deprecated and ignored; retained for backward compatibility. Will be removed in a future release.
            use_l1_small_for_semaphores (bool, optional): Deprecated and ignored; retained for backward compatibility. Will be removed in a future release.

        Returns:
            ttnn.Tensor: The gathered tensor, with output_shape = input_shape for all the unspecified dimensions, and output_shape[dim] = input_shape[dim] * num_devices, where num_devices is the number of devices along the `cluster_axis` if specified, else the total number of devices in the mesh.

        Example:
            >>> full_tensor = torch.randn([1, 1, 32, 256], dtype=torch.bfloat16)
            >>> mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
            >>> ttnn_tensor = ttnn.from_torch(
                            full_tensor,
                            dtype=input_dtype,
                            device=mesh_device,
                            layout=layout,
                            memory_config=mem_config,
                            mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=(1, 8), dims=(-1, -2)))
            >>> output = ttnn.all_gather(ttnn_tensor, dim=0)
            >>> print(output.shape)
            [8, 1, 32, 256]
        )doc";

    ttnn::bind_function<"all_gather">(
        mod,
        doc,
        static_cast<AllGatherFn>(&ttnn::all_gather),
        nb::arg("input_tensor").noconvert(),
        nb::arg("dim"),
        nb::kw_only(),
        nb::arg("cluster_axis") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none(),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("sub_core_grids") = nb::none(),
        // below args are deprecated
        nb::arg("num_links") = nb::none(),
        nb::arg("topology") = nb::none(),
        nb::arg("chunks_per_sync") = nb::none(),
        nb::arg("num_workers_per_link") = nb::none(),
        nb::arg("num_buffers_per_channel") = nb::none(),
        nb::arg("use_l1_small_for_semaphores") = false);
}

}  // namespace ttnn::operations::ccl
