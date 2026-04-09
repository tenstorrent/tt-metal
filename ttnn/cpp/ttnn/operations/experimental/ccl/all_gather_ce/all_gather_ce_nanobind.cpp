// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_ce_nanobind.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_ce/all_gather_ce.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

ttnn::Tensor all_gather_ce_wrapper_sub_core_grids(
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    std::optional<uint32_t> num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    bool use_optimal_ccl_for_llama,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const std::optional<CoreRangeSet>& sub_core_grids,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel) {
    return ttnn::experimental::all_gather_ce(
        input_tensor,
        dim,
        multi_device_global_semaphore,
        num_links,
        memory_config,
        topology,
        subdevice_id,
        use_optimal_ccl_for_llama,
        barrier_semaphore,
        false,
        sub_core_grids,
        num_workers_per_link,
        num_buffers_per_channel);
}

ttnn::Tensor all_gather_ce_wrapper_persistent_buffer(
    const ttnn::Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    std::optional<uint32_t> num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<uint32_t> cluster_axis,
    bool use_optimal_ccl_for_llama,
    bool use_broadcast,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::experimental::all_gather_ce(
        input_tensor,
        persistent_output_buffer,
        dim,
        multi_device_global_semaphore,
        num_links,
        memory_config,
        topology,
        subdevice_id,
        cluster_axis,
        use_optimal_ccl_for_llama,
        barrier_semaphore,
        use_broadcast,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel,
        false,
        sub_core_grids);
}

ttnn::Tensor all_gather_ce_wrapper_mesh_device(
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    const std::optional<size_t> num_preferred_links,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    bool use_optimal_ccl_for_llama,
    bool use_broadcast,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const std::optional<CoreRangeSet>& sub_core_grids,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel) {
    return ttnn::experimental::all_gather_ce(
        input_tensor,
        dim,
        cluster_axis,
        mesh_device,
        topology,
        multi_device_global_semaphore,
        persistent_output_tensor,
        memory_config,
        num_preferred_links,
        subdevice_id,
        use_optimal_ccl_for_llama,
        use_broadcast,
        barrier_semaphore,
        false,
        sub_core_grids,
        num_workers_per_link,
        num_buffers_per_channel);
}

}  // namespace

void bind_all_gather_ce(nb::module_& mod) {
    const auto* doc = R"doc(
        Experimental fork of :func:`ttnn.experimental.all_gather_async` using a separate device operation and kernel
        paths under ``all_gather_ce/device/kernels/``. Same host routing (including composite gather when applicable).
        )doc";

    ttnn::bind_function<"all_gather_ce", "ttnn.experimental.">(
        mod,
        doc,
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                int32_t,
                const std::vector<GlobalSemaphore>&,
                std::optional<uint32_t>,
                const std::optional<ttnn::MemoryConfig>&,
                ttnn::ccl::Topology,
                std::optional<tt::tt_metal::SubDeviceId>,
                bool,
                const std::optional<GlobalSemaphore>&,
                const std::optional<CoreRangeSet>&,
                std::optional<uint32_t>,
                std::optional<uint32_t>>(all_gather_ce_wrapper_sub_core_grids),
            nb::arg("input_tensor"),
            nb::arg("dim"),
            nb::arg("multi_device_global_semaphore"),
            nb::kw_only(),
            nb::arg("num_links") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("topology") = nb::cast(ttnn::ccl::Topology::Ring),
            nb::arg("subdevice_id") = nb::none(),
            nb::arg("use_optimal_ccl_for_llama") = false,
            nb::arg("barrier_semaphore") = nb::none(),
            nb::arg("sub_core_grids") = nb::none(),
            nb::arg("num_workers_per_link") = nb::none(),
            nb::arg("num_buffers_per_channel") = nb::none()),
        ttnn::overload_t(
            &all_gather_ce_wrapper_persistent_buffer,
            nb::arg("input_tensor"),
            nb::arg("persistent_output_buffer") = nb::none(),
            nb::arg("dim"),
            nb::arg("multi_device_global_semaphore"),
            nb::kw_only(),
            nb::arg("num_links") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("topology") = nb::cast(ttnn::ccl::Topology::Ring),
            nb::arg("subdevice_id") = nb::none(),
            nb::arg("cluster_axis") = nb::none(),
            nb::arg("use_optimal_ccl_for_llama") = false,
            nb::arg("use_broadcast") = false,
            nb::arg("barrier_semaphore") = nb::none(),
            nb::arg("chunks_per_sync") = nb::none(),
            nb::arg("num_workers_per_link") = nb::none(),
            nb::arg("num_buffers_per_channel") = nb::none(),
            nb::arg("sub_core_grids") = std::nullopt),
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                int32_t,
                uint32_t,
                const MeshDevice&,
                ttnn::ccl::Topology,
                const std::vector<GlobalSemaphore>&,
                const std::optional<ttnn::Tensor>&,
                const std::optional<size_t>,
                const std::optional<MemoryConfig>&,
                std::optional<tt::tt_metal::SubDeviceId>,
                bool,
                bool,
                const std::optional<GlobalSemaphore>&,
                const std::optional<CoreRangeSet>&,
                std::optional<uint32_t>,
                std::optional<uint32_t>>(all_gather_ce_wrapper_mesh_device),
            nb::arg("input_tensor"),
            nb::arg("dim"),
            nb::arg("cluster_axis"),
            nb::arg("mesh_device"),
            nb::arg("topology"),
            nb::arg("multi_device_global_semaphore"),
            nb::kw_only(),
            nb::arg("persistent_output_tensor") = nb::none(),
            nb::arg("num_preferred_links") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("subdevice_id") = nb::none(),
            nb::arg("use_optimal_ccl_for_llama") = false,
            nb::arg("use_broadcast") = false,
            nb::arg("barrier_semaphore") = nb::none(),
            nb::arg("sub_core_grids") = nb::none(),
            nb::arg("num_workers_per_link") = nb::none(),
            nb::arg("num_buffers_per_channel") = nb::none()));
}

}  // namespace ttnn::operations::experimental::ccl
