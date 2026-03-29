// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "strided_reduce_scatter_async_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/strided_reduce_scatter_async/strided_reduce_scatter_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_strided_reduce_scatter_async(nb::module_& mod) {
    ttnn::bind_function<"strided_reduce_scatter_async", "ttnn.experimental.">(
        mod,
        R"doc(
        Performs a strided reduce-scatter operation on multi-device :attr:`input_tensor` across all devices.
        This variant uses a strided access pattern optimized for matmul output layouts.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor.
            dim (int): Dimension to scatter.
            multi_device_global_semaphore (List[GlobalSemaphore]): Semaphores for synchronization.

        Keyword Args:
            persistent_output_buffers (Optional[List[ttnn.Tensor]]): Pre-allocated output buffers.
            barrier_semaphore (Optional[GlobalSemaphore]): Barrier semaphore for synchronization.
            num_links (int, optional): Number of links to use. Defaults to 1.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for output.
            intermediate_memory_config (ttnn.MemoryConfig, optional): Memory configuration for intermediate tensor.
            topology (ttnn.Topology, optional): Topology (only Ring is supported). Defaults to Ring.
            subdevice_id (Optional[SubDeviceId]): Sub-device identifier.
            cluster_axis (Optional[int]): Cluster axis for multi-device operation.
            chunks_per_sync (Optional[int]): Number of chunks between synchronizations.
            num_workers_per_link (Optional[int]): Number of workers per link.
            num_buffers_per_channel (Optional[int]): Number of buffers per channel.
            mm_cores_y (Optional[int]): Number of cores in Y direction for matmul output layout.
            mm_block_ht (int): Matmul block height in tiles.
            mm_block_wt (int): Matmul block width in tiles.
            mm_N_full_block_wt (Optional[int]): Matmul N block width in tiles.
            chunk_width_in_mm_blocks (Optional[int]): Chunk width in matmul blocks.

        Returns:
            ttnn.Tensor: the output tensor.
        )doc",
        &ttnn::experimental::strided_reduce_scatter_async,
        nb::arg("input_tensor"),
        nb::arg("persistent_output_buffers") = nb::none(),
        nb::arg("dim"),
        nb::arg("multi_device_global_semaphore"),
        nb::kw_only(),
        nb::arg("mm_block_ht"),
        nb::arg("mm_block_wt"),
        nb::arg("barrier_semaphore") = nb::none(),
        nb::arg("num_links") = 1,
        nb::arg("memory_config") = nb::none(),
        nb::arg("intermediate_memory_config") = nb::none(),
        nb::arg("topology") = nb::cast(ttnn::ccl::Topology::Ring),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("cluster_axis") = nb::none(),
        nb::arg("chunks_per_sync") = nb::none(),
        nb::arg("num_workers_per_link") = nb::none(),
        nb::arg("num_buffers_per_channel") = nb::none(),
        nb::arg("mm_cores_y") = nb::none(),
        nb::arg("mm_N_full_block_wt") = nb::none(),
        nb::arg("chunk_width_in_mm_blocks") = nb::none());
}

}  // namespace ttnn::operations::experimental::ccl
