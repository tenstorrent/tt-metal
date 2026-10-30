// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sliding_ring_sdpa_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "sliding_ring_sdpa.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn::experimental::gemma4 {

namespace {
ttnn::Tensor sliding_ring_sdpa_wrapper(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    ttnn::Tensor& persistent_output_buffer_k,
    ttnn::Tensor& persistent_output_buffer_v,
    std::size_t logical_n,
    const ttnn::operations::transformer::SDPAProgramConfig& program_config,
    uint32_t sliding_window_size,
    std::optional<float> scale,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    CoreCoord ccl_core_grid_offset,
    bool use_column_major_ccl,
    std::optional<uint32_t> kv_cache_batch_idx,
    std::optional<uint32_t> kv_actual_isl,
    const std::optional<ttnn::Tensor>& slot_id,
    const std::optional<ttnn::Tensor>& kv_actual_isl_tensor,
    std::optional<uint32_t> kv_cache_num_layers,
    std::optional<uint32_t> kv_cache_layer_idx) {
    return sliding_ring_sdpa(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        persistent_output_buffer_k,
        persistent_output_buffer_v,
        logical_n,
        program_config,
        sliding_window_size,
        dim,
        multi_device_global_semaphore,
        num_links,
        cluster_axis,
        mesh_device,
        topology,
        ccl_core_grid_offset,
        subdevice_id,
        scale,
        compute_kernel_config,
        use_column_major_ccl ? ttnn::ccl::CoreAllocationStrategy::COL_MAJOR
                             : ttnn::ccl::CoreAllocationStrategy::ROW_MAJOR,
        kv_cache_batch_idx,
        kv_actual_isl,
        slot_id,
        kv_actual_isl_tensor,
        kv_cache_num_layers,
        kv_cache_layer_idx);
}
}  // namespace

void bind_sliding_ring_sdpa(nb::module_& mod) {
    auto gemma4 = mod.def_submodule("gemma4", "Gemma4-specific operations");

    const auto* const doc = R"doc(
        Causal sliding-window ring SDPA for Gemma4's sliding layers.

        A Gemma4-owned fork of ttnn.transformer.ring_joint_scaled_dot_product_attention's sliding path:
        chunked prefill over separate K and V caches, block-cyclic Q mapping, the trace-safe slot/prefix
        metadata path, and the fused neighbor-halo exchange that brings each device the previous
        sliding_window_size keys from its ring predecessors. Always causal and unbalanced, with no joint
        tensors. Semantics of the shared arguments match ring_joint_scaled_dot_product_attention.

        Args:
            input_tensor_q (ttnn.Tensor): Queries [b x nh x N/num_devices x dh], the latest chunk.
            input_tensor_k (ttnn.Tensor): K cache [b x nkv x N/num_devices x dh].
            input_tensor_v (ttnn.Tensor): V cache [b x nkv x N/num_devices x dh].

        Keyword args:
            persistent_output_buffer_k (ttnn.Tensor): Compact halo buffer for K.
            persistent_output_buffer_v (ttnn.Tensor): Compact halo buffer for V.
            logical_n (int): Logical global sequence length before sharding.
            program_config (ttnn.SDPAProgramConfig): Program configuration.
            sliding_window_size (int): Sliding window in tokens.
            scale (float, optional): Scale factor for QK^T. Defaults to None.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to None.
            dim (int): Sequence dimension of the ring exchange.
            multi_device_global_semaphore (List[ttnn.GlobalSemaphore]): CCL semaphores.
            num_links (int): Number of fabric links.
            cluster_axis (int): Mesh axis of the ring.
            mesh_device (ttnn.MeshDevice): Multi-device mesh.
            topology (ttnn.Topology): Fabric topology.
            subdevice_id (ttnn.SubDeviceId, optional): Defaults to None.
            ccl_core_grid_offset (ttnn.CoreCoord): Core-grid offset of the CCL workers.
            use_column_major_ccl (bool): Allocate CCL workers column-major. Defaults to False.
            kv_cache_batch_idx (int, optional): Host cache slot. Defaults to None.
            kv_actual_isl (int, optional): Host prior valid KV length. Defaults to None.
            slot_id (ttnn.Tensor, optional): Cache slot read on-device (trace replay). Defaults to None.
            kv_actual_isl_tensor (ttnn.Tensor, optional): Prior valid KV length read on-device. Defaults to None.
            kv_cache_num_layers (int, optional): Layers packed per cache slot. None uses 1.
            kv_cache_layer_idx (int, optional): Layer within the cache slot. None uses 0.

        Returns:
            ttnn.Tensor: The attention output for Q's shape.
        )doc";

    ttnn::bind_function<"sliding_ring_sdpa", "ttnn.experimental.gemma4.">(
        gemma4,
        doc,
        &sliding_ring_sdpa_wrapper,
        nb::arg("input_tensor_q").noconvert(),
        nb::arg("input_tensor_k").noconvert(),
        nb::arg("input_tensor_v").noconvert(),
        nb::kw_only(),
        nb::arg("persistent_output_buffer_k").noconvert(),
        nb::arg("persistent_output_buffer_v").noconvert(),
        nb::arg("logical_n"),
        nb::arg("program_config").noconvert(),
        nb::arg("sliding_window_size"),
        nb::arg("scale") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("dim"),
        nb::arg("multi_device_global_semaphore"),
        nb::arg("num_links"),
        nb::arg("cluster_axis"),
        nb::arg("mesh_device"),
        nb::arg("topology"),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("ccl_core_grid_offset"),
        nb::arg("use_column_major_ccl") = false,
        nb::arg("kv_cache_batch_idx").noconvert() = nb::none(),
        nb::arg("kv_actual_isl").noconvert() = nb::none(),
        nb::arg("slot_id").noconvert() = nb::none(),
        nb::arg("kv_actual_isl_tensor").noconvert() = nb::none(),
        nb::arg("kv_cache_num_layers").noconvert() = nb::none(),
        nb::arg("kv_cache_layer_idx").noconvert() = nb::none());
}

}  // namespace ttnn::experimental::gemma4
