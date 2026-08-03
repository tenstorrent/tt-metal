// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_minimal_async_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/reduce_scatter_minimal_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_reduce_scatter_minimal_async(nb::module_& mod) {
    ttnn::bind_function<"reduce_scatter_minimal_async", "ttnn.experimental.">(
        mod,
        R"doc(
        Performs an reduce-scatter operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor.
            dim (int): Dimension to scatter.
            mesh_device (MeshDevice): Device mesh to perform the line-all-gather operation on.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

        Keyword Args:
            num_links (int, optional): Number of links to use for the reduce-scatter operation. Defaults to the maximum available.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Valid options are Ring and Linear. Defaults to `ttnn.Topology.Ring`.

        Intermediate staging layouts:
            On the ring path (Ring topology, scatter dim != 0) the op supports two layouts for its
            intermediate buffer, and picks between them from the buffer it is handed:

            * Contiguous (chunk-paged). The intermediate is a row-major, interleaved-DRAM staging
              tensor whose page holds a whole chunk, so a chunk's tiles are contiguous at the
              destination. The writer sends a chunk as one or more fused-unicast writes (one per
              fabric packet, since a chunk may exceed a single packet's payload) instead of
              scatter-writing tile by tile, and the reader reads it back in one coalesced
              transaction instead of one per tile. Requires a companion penult intermediate, which
              the op allocates alongside the intermediate unless you pass persistent buffers; to
              pass your own, allocate both with reduce_scatter_minimal_async_create_intermediate_buffer.
            * Tiled. The intermediate mirrors the input tensor's shape and tiled addressing, one tile
              per page. This is the only layout available for Linear topology or scatter dim 0.

            Selection rule:

            * No persistent intermediate passed in persistent_output_buffers: the op allocates the
              contiguous staging buffer itself and uses the contiguous path.
            * A persistent intermediate passed in: whichever layout its TensorSpec matches. A buffer
              matching neither layout is rejected with an error naming both.

            The contiguous path mainly helps small datatypes. With bfloat8_b a tile is 1088 bytes, so
            tile-granular DRAM traffic sits below the NoC-to-DRAM transaction-size knee (~2 KB) and
            leaves bandwidth on the table; coalescing a whole chunk into one transaction roughly
            doubles the intermediate readback throughput. Wider datatypes (bfloat16, float32) already
            clear that knee per tile, so the gain there is smaller. Prefer the contiguous path unless
            you have an existing input-shaped persistent buffer you need to keep using.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:

        )doc",
        &ttnn::experimental::reduce_scatter_minimal_async,
        nb::arg("input_tensor"),
        nb::arg("persistent_output_buffers") = nb::none(),
        nb::arg("dim"),
        nb::arg("multi_device_global_semaphore"),
        nb::kw_only(),
        nb::arg("barrier_semaphore") = nb::none(),
        nb::arg("num_links") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("intermediate_memory_config") = nb::none(),
        nb::arg("topology") = nb::cast(ttnn::ccl::Topology::Ring),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("cluster_axis") = nb::none(),
        nb::arg("chunks_per_sync") = nb::none(),
        nb::arg("num_workers_per_link") = nb::none(),
        nb::arg("num_buffers_per_channel") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());

    ttnn::bind_function<"reduce_scatter_minimal_async_create_intermediate_buffer", "ttnn.experimental.">(
        mod,
        R"doc(
        Allocates the persistent staging buffers for the contiguous ring reduce_scatter_minimal_async
        fast path (Ring topology, scatter dim != 0).

        On this path the intermediate is a chunk-paged, row-major, interleaved-DRAM staging tensor rather
        than an input-shaped tensor, and the 2nd-last ring iteration stages one direction's contribution
        into a second, smaller chunk-paged "penult" intermediate instead of scatter-writing it into the
        tiled output tensor. Both must be allocated with the exact layout the op expects. This helper reuses the
        op's own sizing so the returned tensors are guaranteed to match. Pass the result as
        persistent_output_buffers = [result[0], output_tensor, result[1]] (intermediate at index 0,
        penult intermediate at index 2; output_tensor is the caller's own persistent output). The `dim`, `topology`,
        `cluster_axis`, and `compute_kernel_config` arguments must match those passed to
        reduce_scatter_minimal_async.

        Passing the returned intermediate is what selects the contiguous path; hand
        reduce_scatter_minimal_async an input-shaped intermediate instead and it uses the tiled layout.
        See the reduce_scatter_minimal_async docstring for the full selection rule and for when the
        contiguous path is worth it (chiefly small datatypes such as bfloat8_b).

        Raises if the configuration cannot use the contiguous path at all (Linear topology, or scatter
        dim 0); there the intermediate has the input tensor's shape, can be allocated directly, and
        needs no penult intermediate.

        Returns:
            List[ttnn.Tensor]: [intermediate_buffer, penult_intermediate_buffer], both allocated on the input
            tensor's device.
        )doc",
        &ttnn::experimental::reduce_scatter_minimal_async_create_intermediate_buffer,
        nb::arg("input_tensor"),
        nb::arg("dim"),
        nb::kw_only(),
        nb::arg("topology") = nb::cast(ttnn::ccl::Topology::Ring),
        nb::arg("cluster_axis") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::ccl
