// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "sdpa.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn::operations::transformer {

namespace {
// Special wrapper only for chunked_scaled_dot_product_attention which needs dispatch logic
ttnn::Tensor chunked_scaled_dot_product_attention_wrapper(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& page_table_tensor,
    const nb::object& chunk_start_idx_arg,
    std::optional<ttnn::Tensor> chunk_start_idx_tensor_opt,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<SDPAProgramConfig>& program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    if (chunk_start_idx_tensor_opt.has_value()) {
        return ttnn::transformer::chunked_scaled_dot_product_attention(
            input_tensor_q,
            input_tensor_k,
            input_tensor_v,
            page_table_tensor,
            chunk_start_idx_tensor_opt.value(),
            scale,
            memory_config,
            program_config,
            compute_kernel_config);
    }
    if (chunk_start_idx_arg.is_none()) {
        throw std::runtime_error(
            "chunk_start_idx (int) is required for legacy chunked SDPA. For flexible path use "
            "chunk_start_idx_tensor=...");
    }
    int64_t chunk_start_idx = nb::cast<int64_t>(chunk_start_idx_arg);
    return ttnn::transformer::chunked_scaled_dot_product_attention(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        page_table_tensor,
        chunk_start_idx,
        scale,
        memory_config,
        program_config,
        compute_kernel_config);
}
}  // namespace

void bind_sdpa(nb::module_& mod) {
    const auto* const doc =
        R"doc(
        Causal scaled dot product attention. This API mimicks the PyTorch API of the same name.
        The implementation is FlashAttention-2."

        Accepts a `SDPAProgramConfig` which specifies the grid size and chunk tiles in the Q and K sequence lengths. The op parallelizes over `b`, `nqh`, and Q's `s` dimension.

        Args:
            input_tensor_q (ttnn.Tensor): the input tensor.          [b x nqh x s x dh]
            input_tensor_k (ttnn.Tensor): the input tensor.          [b x nkv x s x dh]
            input_tensor_v (ttnn.Tensor): the input tensor.          [b x nkv x s x dh]

        Keyword args:
            attn_mask (ttnn.Tensor, optional): Defaults to `None`. Shape [b x nqh x s x s] where batch and head dims can each be 1 for broadcasting.
            is_causal (bool): Defaults to `true`.
            scale (float, optional): Defaults to `None`.
            sliding_window_size (int, optional): Defaults to `None`. Size of sliding window for attention. If provided && is_causal, only attends to the last `sliding_window_size` tokens. If provided && !is_causal, attends to a window of size `sliding_window_size` centered at the current position.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            program_config (SDPAProgramConfig, optional): Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.
            attention_sink (ttnn.Tensor, optional): Defaults to `None`. [1 x nqh x 1 x 1]. Single attention sink value per head. The kernel will efficiently replicate this value across all query positions.


        Returns:
            ttnn.Tensor: the output tensor [b x nqh x s x dh].

        )doc";

    ttnn::bind_function<"scaled_dot_product_attention">(
        mod,
        doc,
        ttnn::overload_t(
            &scaled_dot_product_attention_wrapper,
            nb::arg("input_tensor_q").noconvert(),
            nb::arg("input_tensor_k").noconvert(),
            nb::arg("input_tensor_v").noconvert(),
            nb::kw_only(),
            nb::arg("attn_mask") = nb::none(),
            nb::arg("is_causal").noconvert() = true,
            nb::arg("scale") = nb::none(),
            nb::arg("sliding_window_size") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("program_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("attention_sink") = nb::none()));

    const auto* const chunked_doc =
        R"doc(
        Chunked causal scaled dot product attention for paged KV cache and long sequences.
        Processes one Q chunk at a time; K/V are provided as paged cache. The page table
        maps virtual block indices to physical blocks. Two calling conventions:

        **Legacy (chunk_start_idx as int):**
        Pass ``chunk_start_idx`` (integer). The offset is fixed at dispatch time. Use when
        iterating chunks from Python and passing a new scalar each call. Program is cached
        per (config, chunk_start_idx) for the first chunk; later chunks reuse when possible.

        **Flexible (chunk_start_idx_tensor):**
        Pass ``chunk_start_idx_tensor`` (ttnn.Tensor of shape [1], dtype int32) on device.
        The kernel reads the start index from device memory at runtime. Use for:

        - Trace capture/replay: capture one SDPA call, then replay with different
          chunk_start_idx by updating the tensor on device (no recompile).
          One program handles variable prefix lengths by updating the tensor each step.

        The program is compiled once (fixed max page table size); the trace key does not
        include the runtime offset.

        Args:
            input_tensor_q (ttnn.Tensor): Q chunk.          [b x nqh x chunk_s x dh]
            input_tensor_k (ttnn.Tensor): Paged K cache.    [max_blocks x nkv x block_s x dh]
            input_tensor_v (ttnn.Tensor): Paged V cache.    [max_blocks x nkv x block_s x dh]
            page_table_tensor (ttnn.Tensor): Page table.    [b x num_pages], int32.
            chunk_start_idx (int, optional): Legacy: absolute sequence index for this chunk.
                Must be a multiple of program_config.q_chunk_size.
                Must be a multiple of program_config.k_chunk_size (workaround for https://github.com/tenstorrent/tt-metal/issues/35225)
                Omit when using chunk_start_idx_tensor.
            chunk_start_idx_tensor (ttnn.Tensor, optional): Flexible: device tensor [1] int32
                holding the chunk start index; read at runtime. Use for trace or prefix caching.
                Must be a multiple of program_config.q_chunk_size.
                Must be a multiple of program_config.k_chunk_size (workaround for https://github.com/tenstorrent/tt-metal/issues/35225)

        Keyword args:
            scale (float, optional): Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            program_config (SDPAProgramConfig, optional): Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor [b x nqh x s x dh].

        )doc";

    ttnn::bind_function<"chunked_scaled_dot_product_attention">(
        mod,
        chunked_doc,
        ttnn::overload_t(
            &chunked_scaled_dot_product_attention_wrapper,
            nb::arg("input_tensor_q").noconvert(),
            nb::arg("input_tensor_k").noconvert(),
            nb::arg("input_tensor_v").noconvert(),
            nb::arg("page_table_tensor").noconvert(),
            nb::arg("chunk_start_idx") = nb::none(),
            nb::kw_only(),
            nb::arg("chunk_start_idx_tensor") = nb::none(),
            nb::arg("scale").noconvert() = nb::none(),
            nb::arg("memory_config").noconvert() = nb::none(),
            nb::arg("program_config").noconvert() = nb::none(),
            nb::arg("compute_kernel_config").noconvert() = nb::none()));

    const auto* const joint_doc = R"doc(
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

        Returns:
            (ttnn.Tensor, ttnn.Tensor):
              - The attention output for the original Q/K/V shape [b x nh x N x dh].
              - The attention output for the joint Q/K/V shape    [b x nh x L x dh].
        )doc";

    ttnn::bind_function<"joint_scaled_dot_product_attention">(
        mod,
        joint_doc,
        ttnn::overload_t(
            &joint_scaled_dot_product_attention_wrapper,
            nb::arg("input_tensor_q").noconvert(),
            nb::arg("input_tensor_k").noconvert(),
            nb::arg("input_tensor_v").noconvert(),
            nb::arg("joint_tensor_q").noconvert(),
            nb::arg("joint_tensor_k").noconvert(),
            nb::arg("joint_tensor_v").noconvert(),
            nb::kw_only(),
            nb::arg("joint_strategy"),
            nb::arg("program_config").noconvert(),
            nb::arg("scale").noconvert() = nb::none(),
            nb::arg("compute_kernel_config").noconvert() = nb::none()));

    const auto* const ring_joint_doc = R"doc(
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

        Returns:
            (ttnn.Tensor, ttnn.Tensor, ttnn.Tensor):
              - The attention output for the original Q/K/V shape [b x nh x N/num_devices x dh].
              - The attention output for the joint Q/K/V shape    [b x nh x L x dh].
              - The final log-sum-exp of the operation.           [b x nh x (N/num_devices + L) x 1]
        )doc";

    ttnn::bind_function<"ring_joint_scaled_dot_product_attention">(
        mod,
        ring_joint_doc,
        ttnn::overload_t(
            &ring_joint_scaled_dot_product_attention_wrapper,
            nb::arg("input_tensor_q").noconvert(),
            nb::arg("input_tensor_k").noconvert(),
            nb::arg("input_tensor_v").noconvert(),
            nb::arg("joint_tensor_q").noconvert(),
            nb::arg("joint_tensor_k").noconvert(),
            nb::arg("joint_tensor_v").noconvert(),
            nb::kw_only(),
            nb::arg("persistent_output_buffer_k").noconvert(),
            nb::arg("persistent_output_buffer_v").noconvert(),
            nb::arg("joint_strategy"),
            nb::arg("logical_n"),
            nb::arg("program_config").noconvert(),
            nb::arg("scale") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("dim"),
            nb::arg("multi_device_global_semaphore"),
            nb::arg("num_links"),
            nb::arg("cluster_axis"),
            nb::arg("mesh_device"),
            nb::arg("topology"),
            nb::arg("subdevice_id") = nb::none(),
            nb::arg("ccl_core_grid_offset")));

    const auto* const mla_doc =
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
            scale (float, optional): Defaults to `None`.
            program_config (SDPAProgramConfig, optional): Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor [b x nqh x s x dh].

        )doc";

    ttnn::bind_function<"flash_mla_prefill">(
        mod,
        mla_doc,
        ttnn::overload_t(
            &flash_mla_prefill_wrapper,
            nb::arg("input_tensor_q").noconvert(),
            nb::arg("input_tensor_k").noconvert(),
            nb::arg("head_dim_v").noconvert(),
            nb::kw_only(),
            nb::arg("attn_mask") = nb::none(),
            nb::arg("is_causal").noconvert() = true,
            nb::arg("scale") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("program_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()));

    const auto* const chunked_mla_doc =
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
                Must be a multiple of program_config.q_chunk_size.
            head_dim_v (uint32_t): the head dimension of V.

        Keyword args:
            scale (float, optional): Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            program_config (SDPAProgramConfig, optional): Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor [b x nqh x s x dh].

        )doc";

    ttnn::bind_function<"chunked_flash_mla_prefill">(
        mod,
        chunked_mla_doc,
        ttnn::overload_t(
            &chunked_flash_mla_prefill_wrapper,
            nb::arg("input_tensor_q").noconvert(),
            nb::arg("input_tensor_k").noconvert(),
            nb::arg("head_dim_v").noconvert(),
            nb::arg("page_table_tensor").noconvert(),
            nb::arg("chunk_start_idx"),
            nb::kw_only(),
            nb::arg("scale") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("program_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()));

    const auto* const ring_distributed_doc =
        R"doc(
        Ring-distributed causal scaled dot product attention for multi-device execution.
        This optimization distributes query computation across multiple devices in a ring topology,
        with each device computing only a subset of queries to reduce redundant computation
        caused by causal masking. Each device gets two query chunks (one early, one late)
        to balance computational load.

        This operation is CAUSAL-ONLY and generates causal masks internally for each device's
        non-contiguous query assignments. Custom attention masks are not supported.

        Note: This operation outputs results contiguously for the device's assigned queries.
        Model-level code must perform all-gather and reshuffling to restore sequence order.

        Args:
            input_tensor_q (ttnn.Tensor): the input tensor.          [b x nqh x s x dh]
                The sequence length 's' must be divisible by 2*ring_size. Additionally, for proper tile alignment,
                's' should be divisible by TILE_HEIGHT * 2 * ring_size (typically 256 for ring_size=4).
            input_tensor_k (ttnn.Tensor): the input tensor.          [b x nkv x s x dh]
                When using paged KV cache (page_table is provided), this represents paged KV cache blocks with shape
                [max_num_blocks x nkv x block_size x dh], where block_size is the page block size.
            input_tensor_v (ttnn.Tensor): the input tensor.          [b x nkv x s x dh]
                When using paged KV cache (page_table is provided), this represents paged KV cache blocks with shape
                [max_num_blocks x nkv x block_size x dh], where block_size is the page block size.
            ring_size (uint32_t): Number of devices in the ring topology.
            ring_id (uint32_t, optional): This device's position in the ring (0 to ring_size-1).
                                         If None, automatically infers from device coordinate. Defaults to `None`.

        Keyword args:
            scale (float, optional): Attention scaling factor. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            program_config (SDPAProgramConfig, optional): Program configuration. Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute kernel configuration. Defaults to `None`.
            page_table (ttnn.Tensor, optional): Page table tensor for paged KV cache access [b x num_pages]. Defaults to `None`.
            chunk_start_idx (int, optional): Absolute position in the sequence where this chunk starts (for prefix caching).
                Must be a multiple of program_config.q_chunk_size. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor with results for this device's assigned queries [b x nqh x local_s x dh].

        )doc";

    ttnn::bind_function<"ring_distributed_scaled_dot_product_attention">(
        mod,
        ring_distributed_doc,
        ttnn::overload_t(
            &ring_distributed_scaled_dot_product_attention_wrapper,
            nb::arg("input_tensor_q").noconvert(),
            nb::arg("input_tensor_k").noconvert(),
            nb::arg("input_tensor_v").noconvert(),
            nb::arg("ring_size").noconvert(),
            nb::arg("ring_id") = nb::none(),
            nb::kw_only(),
            nb::arg("scale") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("program_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("page_table") = nb::none(),
            nb::arg("chunk_start_idx") = nb::none()));
}
}  // namespace ttnn::operations::transformer
