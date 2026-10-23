// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_create_qkv_heads_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <tuple>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "all_reduce_create_qkv_heads.hpp"
#include "ttnn/types.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

namespace ttnn::operations::experimental::transformer::detail {

void bind_all_reduce_create_qkv_heads(nb::module_& mod) {
    ttnn::bind_function<"all_reduce_create_qkv_heads", "ttnn.experimental.">(
        mod,
        R"doc(
        Performs an all_reduce operation on multi-device :attr:`input_tensor` across all devices and creates QKV heads.
        This operation requires a persistent fabric to be enabled in order to function.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor
            buffer_tensor (ttnn.Tensor): buffer tensor for intermediate results
            batch_offset (ttnn.Tensor): Batch offset tensor
            cluster_axis (int): Provided a MeshTensor, the axis corresponding to MeshDevice to perform the operation on
            mesh_device (MeshDevice): Device mesh to perform the operation on
            multi_device_global_semaphore (ttnn.GlobalSemaphore): A single semaphore used for cross-device
                synchronization. The ``multi_device_`` prefix is a legacy name; one ``GlobalSemaphore`` is
                expected here, not a per-device collection.
            num_heads (int): Number of attention heads

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration of the all_reduce output
            topology (ttnn.Topology, optional): The topology configuration (Ring or Linear). Defaults to Linear
            num_links (int, optional): Number of links to use for the operation
            subdevice_id (SubDeviceId, optional): Worker subdevice ID
            num_kv_heads (int, optional): Number of key/value heads
            slice_size (int, optional): Size of slices
            final_memory_config (ttnn.MemoryConfig, optional): Memory configuration of the Q, K and V outputs
            dtype (ttnn.DataType, optional): Data type of the Q, K and V outputs
            use_noc1_only (bool, optional): Restrict the operation to NOC1. Defaults to False

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]: the all_reduce output tensor followed by
            the Query, Key and Value tensors
        )doc",
        &ttnn::experimental::all_reduce_create_qkv_heads,
        nb::arg("input_tensor"),
        nb::arg("buffer_tensor"),
        nb::arg("batch_offset"),
        nb::arg("cluster_axis"),
        nb::arg("mesh_device"),
        nb::arg("multi_device_global_semaphore"),
        nb::arg("num_heads"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("topology") = ttnn::ccl::Topology::Linear,
        nb::arg("num_links") = nb::none(),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("num_kv_heads") = nb::none(),
        nb::arg("slice_size") = nb::none(),
        nb::arg("final_memory_config") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("use_noc1_only") = false);
}

}  // namespace ttnn::operations::experimental::transformer::detail
