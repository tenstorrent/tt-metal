// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_create_qkv_heads_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <tuple>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/all_reduce_create_qkv_heads.hpp"
#include "ttnn/types.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

namespace ttnn::operations::experimental::transformer::detail {

namespace {

template <typename ccl_operation_t>
void bind_all_reduce_create_qkv_heads(nb::module_& mod, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               ttnn::Tensor& buffer_tensor,
               const ttnn::Tensor& batch_offset,
               const uint32_t cluster_axis,
               const MeshDevice& mesh_device,
               const GlobalSemaphore& multi_device_global_semaphore,
               const uint32_t num_heads,
               const ttnn::MemoryConfig& memory_config,
               ttnn::ccl::Topology topology,
               const std::optional<size_t> num_links,
               std::optional<tt::tt_metal::SubDeviceId> worker_subdevice_id_opt,
               std::optional<const uint32_t> num_kv_heads,
               const std::optional<const uint32_t> slice_size,
               const std::optional<MemoryConfig>& final_memory_config,
               const std::optional<const DataType> dtype,
               bool use_noc1_only) -> std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> {
                return self(
                    input_tensor,
                    buffer_tensor,
                    batch_offset,
                    cluster_axis,
                    mesh_device,
                    multi_device_global_semaphore,
                    num_heads,
                    memory_config,
                    topology,
                    num_links,
                    worker_subdevice_id_opt,
                    num_kv_heads,
                    slice_size,
                    final_memory_config,
                    dtype,
                    use_noc1_only);
            },
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
            nb::arg("use_noc1_only") = false});
}

}  // namespace

void bind_all_reduce_create_qkv_heads(nb::module_& mod) {
    bind_all_reduce_create_qkv_heads(
        mod,
        ttnn::experimental::all_reduce_create_qkv_heads,
        R"doc(
        Performs an all_reduce operation on multi-device :attr:`input_tensor` across all devices and creates QKV heads.
        This operation requires a persistent fabric to be enabled in order to function.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor
            buffer_tensor (ttnn.Tensor): buffer tensor for intermediate results
            cluster_axis (int): Provided a MeshTensor, the axis corresponding to MeshDevice to perform the operation on
            mesh_device (MeshDevice): Device mesh to perform the operation on
            multi_device_global_semaphore (MultiDeviceGlobalSemaphore): Semaphore for multi-device synchronization
            num_heads (int): Number of attention heads

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation
            topology (ttnn.Topology, optional): The topology configuration (Ring or Linear). Defaults to Linear
            num_links (int, optional): Number of links to use for the operation
            subdevice_id (SubDeviceId, optional): Worker subdevice ID
            num_kv_heads (int, optional): Number of key/value heads
            overlap_qk_coregrid (bool, optional): Whether to overlap Q and K coregrid. Defaults to True
            batch_offset (Tensor, optional): Batch offset tensor
            slice_size (int, optional): Size of slices
            final_memory_config (MemoryConfig, optional): Final memory configuration
            optional_output_tensors (tuple[Tensor, Tensor, Tensor], optional): Optional pre-allocated output tensors

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]: Query, Key, and Value tensors
        )doc");
}

}  // namespace ttnn::operations::experimental::transformer::detail
