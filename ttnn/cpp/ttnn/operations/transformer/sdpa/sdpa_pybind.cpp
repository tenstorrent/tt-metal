// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sdpa.hpp"
#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn::operations::transformer {

void py_bind_sdpa(py::module& module) {
    auto doc =
        R"doc(
        Causal scaled dot product attention. This API mimicks the PyTorch API of the same name.
        The implementation is FlashAttention-2."

        Accepts a `SDPAProgramConfig` which specifies the grid size and chunk tiles in the Q and K sequence lengths. The op parallelizes over `b`, `nqh`, and Q's `s` dimension.

        Args:
            input_tensor_q (ttnn.Tensor): the input tensor.          [b x nqh x s x dh]
            input_tensor_k (ttnn.Tensor): the input tensor.          [b x nkv x s x dh]
            input_tensor_v (ttnn.Tensor): the input tensor.          [b x nkv x s x dh]

        Keyword args:
            attn_mask (ttnn.Tensor, optional): Defaults to `None`. [b x 1 x s x s]. Head broadcasting is implied.
            is_causal (bool): Defaults to `true`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.
            scale (float, optional): Defaults to `None`.
            program_config (SDPAProgramConfig, optional): Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor [b x nqh x s x dh].

        )doc";

    using OperationType = decltype(ttnn::transformer::scaled_dot_product_attention);
    ttnn::bind_registered_operation(
        module,
        ttnn::transformer::scaled_dot_product_attention,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor_q,
               const ttnn::Tensor& input_tensor_k,
               const ttnn::Tensor& input_tensor_v,
               std::optional<ttnn::Tensor> attn_mask,
               bool is_causal,
               std::optional<float> scale,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<SDPAProgramConfig> program_config,
               std::optional<DeviceComputeKernelConfig> compute_kernel_config,
               QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor_q,
                    input_tensor_k,
                    input_tensor_v,
                    attn_mask,
                    is_causal,
                    scale,
                    memory_config,
                    program_config,
                    compute_kernel_config);
            },
            py::arg("input_tensor_q").noconvert(),
            py::arg("input_tensor_k").noconvert(),
            py::arg("input_tensor_v").noconvert(),
            py::kw_only(),
            py::arg("attn_mask").noconvert() = std::nullopt,
            py::arg("is_causal").noconvert() = true,
            py::arg("scale").noconvert() = std::nullopt,
            py::arg("memory_config").noconvert() = std::nullopt,
            py::arg("program_config").noconvert() = std::nullopt,
            py::arg("compute_kernel_config").noconvert() = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });

    auto chunked_doc =
        R"doc(
        Chunked causal scaled dot product attention for processing long sequences in chunks.
        This variant allows processing of sequences longer than the maximum supported length
        by splitting the input into chunks and maintaining KV cache state.
        The KV cache is page-based, and the page table tensor is used to map the page indices to the corresponding KV cache indices.

        Args:
            input_tensor_q (ttnn.Tensor): the input tensor.          [b x nqh x s x dh]
            input_tensor_k (ttnn.Tensor): the input tensor.          [b x nkv x s x dh]
            input_tensor_v (ttnn.Tensor): the input tensor.          [b x nkv x s x dh]
            page_table_tensor (ttnn.Tensor): the page table tensor.  [b x num_pages]
            chunk_start_idx (int): Absolute position in the sequence where this chunk starts.

        Keyword args:
            scale (float, optional): Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            program_config (SDPAProgramConfig, optional): Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor [b x nqh x s x dh].

        )doc";

    using ChunkedOperationType = decltype(ttnn::transformer::chunked_scaled_dot_product_attention);
    ttnn::bind_registered_operation(
        module,
        ttnn::transformer::chunked_scaled_dot_product_attention,
        chunked_doc,
        ttnn::pybind_overload_t{
            [](const ChunkedOperationType& self,
               const ttnn::Tensor& input_tensor_q,
               const ttnn::Tensor& input_tensor_k,
               const ttnn::Tensor& input_tensor_v,
               const ttnn::Tensor& page_table_tensor,
               int64_t chunk_start_idx,
               std::optional<float> scale,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<SDPAProgramConfig> program_config,
               std::optional<DeviceComputeKernelConfig> compute_kernel_config,
               QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor_q,
                    input_tensor_k,
                    input_tensor_v,
                    page_table_tensor,
                    chunk_start_idx,
                    scale,
                    memory_config,
                    program_config,
                    compute_kernel_config);
            },
            py::arg("input_tensor_q").noconvert(),
            py::arg("input_tensor_k").noconvert(),
            py::arg("input_tensor_v").noconvert(),
            py::arg("page_table_tensor").noconvert(),
            py::arg("chunk_start_idx"),
            py::kw_only(),
            py::arg("scale").noconvert() = std::nullopt,
            py::arg("memory_config").noconvert() = std::nullopt,
            py::arg("program_config").noconvert() = std::nullopt,
            py::arg("compute_kernel_config").noconvert() = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });

    auto joint_doc = R"doc(
        JointAttention operation that efficiently performs non-causal attention over two
        sets of query, key, and value tensors. Internally, these are concatenated in the sequence
        dimension (joint_strategy = "rear"), then attention is computed once. The
        output is split ("sliced") into two parts: one for the original Q/K/V chunk,
        and one for the joint Q/K/V chunk.

        This op handles optional padding via an attention mask to omit padded tokens from
        both the "original" and "joint" sequences.

        Args:
            input_tensor_q (ttnn.Tensor): Original queries  [b x nh x N x dh].
            input_tensor_k (ttnn.Tensor): Original keys     [b x nh x N x dh].
            input_tensor_v (ttnn.Tensor): Original values   [b x nh x N x dh].

            joint_tensor_q (ttnn.Tensor): Joint queries     [b x nh x L x dh].
            joint_tensor_k (ttnn.Tensor): Joint keys        [b x nh x L x dh].
            joint_tensor_v (ttnn.Tensor): Joint values      [b x nh x L x dh].

        Keyword args:
            joint_strategy (str): Strategy for joint attention. Must be "rear".
            program_config (ttnn.SDPAProgramConfig)
            scale (float, optional): Scale factor for QK^T. Defaults to None.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional):Defaults to None.
            queue_id (int, optional): Command queue ID. Defaults to 0.

        Returns:
            (ttnn.Tensor, ttnn.Tensor):
              - The attention output for the original Q/K/V shape [b x nh x N x dh].
              - The attention output for the joint Q/K/V shape    [b x nh x L x dh].
        )doc";

    using JointOperationType = decltype(ttnn::transformer::joint_scaled_dot_product_attention);

    ttnn::bind_registered_operation(
        module,
        ttnn::transformer::joint_scaled_dot_product_attention,
        joint_doc,
        ttnn::pybind_overload_t{
            [](const JointOperationType& self,
               const ttnn::Tensor& input_tensor_q,
               const ttnn::Tensor& input_tensor_k,
               const ttnn::Tensor& input_tensor_v,
               const ttnn::Tensor& joint_tensor_q,
               const ttnn::Tensor& joint_tensor_k,
               const ttnn::Tensor& joint_tensor_v,
               const std::string& joint_strategy,
               SDPAProgramConfig program_config,
               std::optional<float> scale,
               std::optional<DeviceComputeKernelConfig> compute_kernel_config,
               QueueId queue_id) {
                auto outputs = self(
                    queue_id,
                    input_tensor_q,
                    input_tensor_k,
                    input_tensor_v,
                    joint_tensor_q,
                    joint_tensor_k,
                    joint_tensor_v,
                    joint_strategy,
                    program_config,
                    scale,
                    compute_kernel_config);
                return outputs;
            },
            py::arg("input_tensor_q").noconvert(),
            py::arg("input_tensor_k").noconvert(),
            py::arg("input_tensor_v").noconvert(),
            py::arg("joint_tensor_q").noconvert(),
            py::arg("joint_tensor_k").noconvert(),
            py::arg("joint_tensor_v").noconvert(),
            py::kw_only(),
            py::arg("joint_strategy"),
            py::arg("program_config").noconvert(),
            py::arg("scale").noconvert() = std::nullopt,
            py::arg("compute_kernel_config").noconvert() = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});

    auto ring_joint_doc = R"doc(
        RingJointAttention operation that efficiently performs non-causal attention over two
        sets of query, key, and value tensors, where the first set is sharded across devices in the sequence dimension.
        Internally, these are concatenated in the sequence dimension (joint_strategy = "rear"),
        then attention is computed once. The output is split ("sliced") into two parts: one for the original Q/K/V chunk,
        and one for the joint Q/K/V chunk.

        This op handles optional padding via an attention mask to omit padded tokens from
        both the "original" and "joint" sequences.

        Since N must be divisible by the number of devices, the logical N must be passed in.

        Args:
            input_tensor_q (ttnn.Tensor): Original queries  [b x nh x N/num_devices x dh].
            input_tensor_k (ttnn.Tensor): Original keys     [b x nh x N/num_devices x dh].
            input_tensor_v (ttnn.Tensor): Original values   [b x nh x N/num_devices x dh].

            joint_tensor_q (ttnn.Tensor): Joint queries     [b x nh x L x dh].
            joint_tensor_k (ttnn.Tensor): Joint keys        [b x nh x L x dh].
            joint_tensor_v (ttnn.Tensor): Joint values      [b x nh x L x dh].

        Keyword args:
            persistent_output_buffer_k (ttnn.Tensor): Persistent buffer for gathered K tensor.
            persistent_output_buffer_v (ttnn.Tensor): Persistent buffer for gathered V tensor.
            joint_strategy (str): Strategy for joint attention. Must be "rear".
            logical_n (int): The logical sequence length N before sharding across devices.
            program_config (ttnn.SDPAProgramConfig): Program configuration for the operation.
            scale (float, optional): Scale factor for QK^T. Defaults to None.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to None.
            dim (int): Dimension along which to perform the ring all-gather operation.
            multi_device_global_semaphore (List[ttnn.GlobalSemaphore]): Global semaphores for multi-device synchronization.
            num_links (int): Number of communication links to use for ring all-gather.
            cluster_axis (int): Axis of the mesh device along which to perform the all-gather.
            mesh_device (ttnn.MeshDevice): Multi-device mesh for distributed computation.
            topology (ttnn.ccl.Topology): Communication topology (Ring or Linear).
            subdevice_id (Optional[tt.tt_metal.SubDeviceId]): Sub-device identifier. Defaults to None.
            ccl_core_grid_offset (ttnn.CoreCoord): Core grid offset for CCL operations.
            queue_id (int, optional): Command queue ID. Defaults to 0.

        Returns:
            (ttnn.Tensor, ttnn.Tensor, ttnn.Tensor):
              - The attention output for the original Q/K/V shape [b x nh x N/num_devices x dh].
              - The attention output for the joint Q/K/V shape    [b x nh x L x dh].
              - The final log-sum-exp of the operation.           [b x nh x (N/num_devices + L) x 1]
        )doc";

    using RingJointOperationType = decltype(ttnn::transformer::ring_joint_scaled_dot_product_attention);

    ttnn::bind_registered_operation(
        module,
        ttnn::transformer::ring_joint_scaled_dot_product_attention,
        ring_joint_doc,
        ttnn::pybind_overload_t{
            [](const RingJointOperationType& self,
               const ttnn::Tensor& input_tensor_q,
               const ttnn::Tensor& input_tensor_k,
               const ttnn::Tensor& input_tensor_v,
               const ttnn::Tensor& joint_tensor_q,
               const ttnn::Tensor& joint_tensor_k,
               const ttnn::Tensor& joint_tensor_v,
               ttnn::Tensor& persistent_output_buffer_k,
               ttnn::Tensor& persistent_output_buffer_v,
               const std::string& joint_strategy,
               std::size_t logical_n,
               SDPAProgramConfig program_config,
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
               QueueId queue_id) {
                auto outputs = self(
                    queue_id,
                    input_tensor_q,
                    input_tensor_k,
                    input_tensor_v,
                    joint_tensor_q,
                    joint_tensor_k,
                    joint_tensor_v,
                    persistent_output_buffer_k,
                    persistent_output_buffer_v,
                    joint_strategy,
                    logical_n,
                    program_config,
                    dim,
                    multi_device_global_semaphore,
                    num_links,
                    cluster_axis,
                    mesh_device,
                    topology,
                    subdevice_id,
                    ccl_core_grid_offset,
                    scale,
                    compute_kernel_config);
                return outputs;
            },
            py::arg("input_tensor_q").noconvert(),
            py::arg("input_tensor_k").noconvert(),
            py::arg("input_tensor_v").noconvert(),
            py::arg("joint_tensor_q").noconvert(),
            py::arg("joint_tensor_k").noconvert(),
            py::arg("joint_tensor_v").noconvert(),
            py::kw_only(),
            py::arg("persistent_output_buffer_k").noconvert(),
            py::arg("persistent_output_buffer_v").noconvert(),
            py::arg("joint_strategy"),
            py::arg("logical_n"),
            py::arg("program_config").noconvert(),
            py::arg("scale").noconvert() = std::nullopt,
            py::arg("compute_kernel_config").noconvert() = std::nullopt,
            py::arg("dim"),
            py::arg("multi_device_global_semaphore"),
            py::arg("num_links"),
            py::arg("cluster_axis"),
            py::arg("mesh_device"),
            py::arg("topology"),
            py::arg("subdevice_id") = std::nullopt,
            py::arg("ccl_core_grid_offset"),
            py::arg("queue_id") = DefaultQueueId});

    auto mla_doc =
        R"doc(
        Causal MLA attention."

        Accepts a `SDPAProgramConfig` which specifies the grid size and chunk tiles in the Q and K sequence lengths. The op parallelizes over `b`, `nqh`, and Q's `s` dimension.

        Args:
            input_tensor_q (ttnn.Tensor): the input tensor.          [b x nqh x s x dh]
            input_tensor_k (ttnn.Tensor): the input tensor.          [b x nkv x s x dh]
            head_dim_v (uint32_t): the head dimension of V.

        Keyword args:
            attn_mask (ttnn.Tensor, optional): Defaults to `None`. [b x 1 x s x s]. Head broadcasting is implied.
            is_causal (bool): Defaults to `true`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.
            scale (float, optional): Defaults to `None`.
            program_config (SDPAProgramConfig, optional): Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor [b x nqh x s x dh].

        )doc";

    using MLAOperationType = decltype(ttnn::transformer::flash_mla_prefill);
    ttnn::bind_registered_operation(
        module,
        ttnn::transformer::flash_mla_prefill,
        mla_doc,
        ttnn::pybind_overload_t{
            [](const MLAOperationType& self,
               const ttnn::Tensor& input_tensor_q,
               const ttnn::Tensor& input_tensor_k,
               const uint32_t head_dim_v,
               std::optional<ttnn::Tensor> attn_mask,
               bool is_causal,
               std::optional<float> scale,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<SDPAProgramConfig> program_config,
               std::optional<DeviceComputeKernelConfig> compute_kernel_config,
               QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor_q,
                    input_tensor_k,
                    head_dim_v,
                    attn_mask,
                    is_causal,
                    scale,
                    memory_config,
                    program_config,
                    compute_kernel_config);
            },
            py::arg("input_tensor_q").noconvert(),
            py::arg("input_tensor_k").noconvert(),
            py::arg("head_dim_v").noconvert(),
            py::kw_only(),
            py::arg("attn_mask").noconvert() = std::nullopt,
            py::arg("is_causal").noconvert() = true,
            py::arg("scale").noconvert() = std::nullopt,
            py::arg("memory_config").noconvert() = std::nullopt,
            py::arg("program_config").noconvert() = std::nullopt,
            py::arg("compute_kernel_config").noconvert() = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });

    auto chunked_mla_doc =
        R"doc(
        Chunked causal scaled dot product attention for processing long sequences in chunks.
        This variant allows processing of sequences longer than the maximum supported length
        by splitting the input into chunks and maintaining KV cache state.
        The KV cache is page-based, and the page table tensor is used to map the page indices to the corresponding KV cache indices.

        Args:
            input_tensor_q (ttnn.Tensor): the input tensor.          [b x nqh x s x dh]
            input_tensor_k (ttnn.Tensor): the input tensor.          [b x nkv x s x dh]
            page_table_tensor (ttnn.Tensor): the page table tensor.  [b x num_pages]
            chunk_start_idx (int): Absolute position in the sequence where this chunk starts.
            head_dim_v (uint32_t): the head dimension of V.

        Keyword args:
            scale (float, optional): Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            program_config (SDPAProgramConfig, optional): Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor [b x nqh x s x dh].

        )doc";

    using MLAChunkedOperationType = decltype(ttnn::transformer::chunked_flash_mla_prefill);
    ttnn::bind_registered_operation(
        module,
        ttnn::transformer::chunked_flash_mla_prefill,
        chunked_mla_doc,
        ttnn::pybind_overload_t{
            [](const MLAChunkedOperationType& self,
               const ttnn::Tensor& input_tensor_q,
               const ttnn::Tensor& input_tensor_k,
               const uint32_t head_dim_v,
               const ttnn::Tensor& page_table_tensor,
               int64_t chunk_start_idx,
               std::optional<float> scale,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<SDPAProgramConfig> program_config,
               std::optional<DeviceComputeKernelConfig> compute_kernel_config,
               QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor_q,
                    input_tensor_k,
                    head_dim_v,
                    page_table_tensor,
                    chunk_start_idx,
                    scale,
                    memory_config,
                    program_config,
                    compute_kernel_config);
            },
            py::arg("input_tensor_q").noconvert(),
            py::arg("input_tensor_k").noconvert(),
            py::arg("head_dim_v").noconvert(),
            py::arg("page_table_tensor").noconvert(),
            py::arg("chunk_start_idx"),
            py::kw_only(),
            py::arg("scale").noconvert() = std::nullopt,
            py::arg("memory_config").noconvert() = std::nullopt,
            py::arg("program_config").noconvert() = std::nullopt,
            py::arg("compute_kernel_config").noconvert() = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });
}
}  // namespace ttnn::operations::transformer
