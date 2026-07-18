// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
#include "sparse_sdpa.hpp"
#include "sparse_sdpa_msa.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::transformer {

namespace {
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> ring_joint_scaled_dot_product_attention_wrapper(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const std::optional<ttnn::Tensor>& joint_tensor_q,
    const std::optional<ttnn::Tensor>& joint_tensor_k,
    const std::optional<ttnn::Tensor>& joint_tensor_v,
    ttnn::Tensor& persistent_output_buffer_k,
    ttnn::Tensor& persistent_output_buffer_v,
    const std::string& joint_strategy,
    std::size_t logical_n,
    const SDPAProgramConfig& program_config,
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
    bool is_causal,
    bool is_balanced,
    bool is_cross,
    std::optional<uint32_t> kv_cache_batch_idx,
    std::optional<uint32_t> kv_actual_isl) {
    auto strategy = use_column_major_ccl ? ttnn::ccl::CoreAllocationStrategy::COL_MAJOR
                                         : ttnn::ccl::CoreAllocationStrategy::ROW_MAJOR;

    auto outputs = ttnn::transformer::ring_joint_scaled_dot_product_attention(
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
        is_causal,
        is_balanced,
        is_cross,
        scale,
        compute_kernel_config,
        strategy,
        kv_cache_batch_idx,
        kv_actual_isl);
    return outputs;
}

std::tuple<ttnn::Tensor, ttnn::Tensor> ring_mla_wrapper(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_kv,
    ttnn::Tensor& persistent_output_buffer_kv,
    uint32_t head_dim_v,
    std::size_t logical_n,
    const SDPAProgramConfig& program_config,
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
    bool is_balanced,
    std::optional<uint32_t> kv_cache_batch_idx,
    std::optional<uint32_t> kv_actual_isl) {
    auto strategy = use_column_major_ccl ? ttnn::ccl::CoreAllocationStrategy::COL_MAJOR
                                         : ttnn::ccl::CoreAllocationStrategy::ROW_MAJOR;
    return ttnn::transformer::ring_mla(
        input_tensor_q,
        input_tensor_kv,
        persistent_output_buffer_kv,
        head_dim_v,
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
        is_balanced,
        scale,
        compute_kernel_config,
        strategy,
        kv_cache_batch_idx,
        kv_actual_isl);
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> exp_ring_joint_scaled_dot_product_attention_wrapper(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const std::optional<ttnn::Tensor>& joint_tensor_q,
    const std::optional<ttnn::Tensor>& joint_tensor_k,
    const std::optional<ttnn::Tensor>& joint_tensor_v,
    ttnn::Tensor& persistent_output_buffer_k,
    ttnn::Tensor& persistent_output_buffer_v,
    const std::string& joint_strategy,
    std::size_t logical_n,
    const SDPAProgramConfig& program_config,
    std::optional<float> scale,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    uint32_t num_workers_per_link,
    uint32_t num_buffers_per_channel) {
    return ttnn::transformer::ExecuteExpRingJointAttention::invoke(
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
        scale,
        compute_kernel_config,
        num_workers_per_link,
        num_buffers_per_channel);
}

}  // namespace

namespace {
ttnn::Tensor flash_mla_prefill_wrapper(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const uint32_t head_dim_v,
    const std::optional<ttnn::Tensor>& attn_mask,
    bool is_causal,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<SDPAProgramConfig>& program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    return ttnn::transformer::flash_mla_prefill(
        input_tensor_q,
        input_tensor_k,
        head_dim_v,
        std::nullopt,
        attn_mask,
        is_causal,
        scale,
        memory_config,
        program_config,
        compute_kernel_config);
}

ttnn::Tensor flash_mla_prefill_wrapper_input_tensor(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const std::optional<ttnn::Tensor>& attn_mask,
    bool is_causal,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<SDPAProgramConfig>& program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    return ttnn::transformer::flash_mla_prefill(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v.logical_shape()[-1],
        input_tensor_v,
        attn_mask,
        is_causal,
        scale,
        memory_config,
        program_config,
        compute_kernel_config);
}

// Dispatch: chunk_start_idx_tensor present → flexible (runtime offset); else legacy (chunk_start_idx int).
// nanobind optional caster converts Python None|int at the wrapper boundary
// (GIL held); the body runs with the GIL released (call_guard applied by
// bind_function) and uses only C++ values.
ttnn::Tensor chunked_scaled_dot_product_attention_wrapper(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& page_table_tensor,
    std::optional<int64_t> chunk_start_idx_arg,
    std::optional<ttnn::Tensor> chunk_start_idx_tensor_opt,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<SDPAProgramConfig>& program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<uint32_t> block_size,
    std::optional<uint32_t> num_kv_heads) {
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
            compute_kernel_config,
            block_size,
            num_kv_heads);
    }
    if (!chunk_start_idx_arg.has_value()) {
        throw std::runtime_error(
            "chunk_start_idx (int) is required for legacy chunked SDPA. For flexible path use "
            "chunk_start_idx_tensor=...");
    }
    return ttnn::transformer::chunked_scaled_dot_product_attention(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        page_table_tensor,
        *chunk_start_idx_arg,
        scale,
        memory_config,
        program_config,
        compute_kernel_config,
        block_size,
        num_kv_heads);
}

}  // namespace

void bind_sdpa(nb::module_& mod) {
    const auto* const doc =
        R"doc(
        Causal scaled dot product attention. This API mimics the PyTorch API of the same name.
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
            cu_window_seqlens (ttnn.Tensor, optional): Defaults to `None`. 1D int32/uint32 ROW_MAJOR tensor of cumulative window boundaries [0, w1, w1+w2, ..., s]. When provided, computes block-diagonal (windowed) attention where each token attends only within its window; the mask is built on-device. Non-causal; mutually exclusive with attn_mask/is_causal/sliding_window_size.


        Returns:
            ttnn.Tensor: the output tensor [b x nqh x s x dh].

        )doc";

    ttnn::bind_function<"scaled_dot_product_attention", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::scaled_dot_product_attention,
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
        nb::arg("attention_sink") = nb::none(),
        nb::arg("cu_window_seqlens") = nb::none());

    ttnn::bind_function<"sparse_sdpa", "ttnn.transformer.">(
        mod,
        R"doc(
        Sparse MLA prefill (DeepSeek DSA), Blackhole single-chip. softmax(Q@Kᵀ·scale masked)@V over the
        top-k selected latents per query token; V = kv[..., :v_dim]. Masking is baked into `indices`
        (0xFFFFFFFF = masked; sentinels are a contiguous tail). Inputs are ROW-MAJOR DRAM-interleaved.
        K_DIM (the q/kv head dim) is taken from the tensors.

        Args:
            q (ttnn.Tensor):       [1, H, S, K_DIM] bf16 or fp8_e4m3 (H a multiple of 32)
            kv (ttnn.Tensor):      [1, 1, T, K_DIM] bf16/raw fp8, or one packed scaled-FP8 row per token.
                                   The packed DSA row is [512 FP8 | 4 FP32 scales | 64 BF16 RoPE] = 656 bytes.
                                   When cache_batch_idx is set, B may exceed 1 and kv may be ND-sharded.
            indices (ttnn.Tensor): [1, 1, S, TOPK] uint32
            v_dim (int):           width of V (leading v_dim cols of the K_DIM-wide cache); the output width.

        Keyword args:
            scale (float, optional): defaults to K_DIM**-0.5.
            k_chunk_size (int): defaults to 128 (must divide TOPK, multiple of 32).
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional).
            cache_batch_idx (int, optional): select the batch slot of a shared [B, 1, T, K_DIM] kv cache.
                It is a dynamic runtime arg, so changing the slot does not recompile. Changing T also reuses the
                program for a plain interleaved cache, but recompiles for sharded or block-cyclic caches.
            block_cyclic_sp_axis (int, optional): when set (with block_cyclic_chunk_local), `indices` are NATURAL
                token positions and kv is stored block-cyclic across an SP-sharded cache; the kernel remaps each
                index natural->physical page on the fly (no host reorder needed). This is the MESH axis the cache
                was striped over: `sp` is read from the mesh shape on that axis (the op derives it, so it cannot
                disagree with the device). T % sp == 0 required. Both must be set together.
            block_cyclic_chunk_local (int, optional): the per-shard chunk length (chunk_size_global / sp).
                Required iff block_cyclic_sp_axis is set. Cross-checked against q's per-chip seq length: must be
                q_isl or tp*q_isl (tp = mesh_size/sp) — the only two values it can legally take.
        Returns:
            ttnn.Tensor: [1, H, S, v_dim] ROW-MAJOR, DRAM interleaved; dtype matches q (bf16->bf16, fp8->fp8).
        )doc",
        &ttnn::transformer::sparse_sdpa,
        nb::arg("q").noconvert(),
        nb::arg("kv").noconvert(),
        nb::arg("indices").noconvert(),
        nb::arg("v_dim"),
        nb::kw_only(),
        nb::arg("scale") = nb::none(),
        nb::arg("k_chunk_size") = 128,
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("cache_batch_idx") = nb::none(),
        nb::arg("block_cyclic_sp_axis") = nb::none(),
        nb::arg("block_cyclic_chunk_local") = nb::none());

    ttnn::bind_function<"sparse_sdpa_msa", "ttnn.transformer.">(
        mod,
        R"doc(
        MSA block-sparse prefill (MiniMax Sparse Attention), Blackhole single-chip.
        Attends the block_size-token K/V blocks named in `indices`; -1 (0xFFFFFFFF) sentinels mask a contiguous
        tail. Block selection bounds causality to block granularity; pass `chunk_start_idx` to additionally
        enforce a token-level causal mask on the diagonal block (required for correct causal prefill). RoPE and
        QK-norm are applied upstream.

        Args:
            q (ttnn.Tensor):       [1, H, S, d] bf16 | fp8_e4m3 ROW_MAJOR.
                                   H must be divisible by n_kv; H/n_kv may be 16 or a multiple of 32.
            k (ttnn.Tensor):       [B, n_kv, T, d] TILE bf16|bfloat8_b (B>1 only when indexed)
            v (ttnn.Tensor):       [B, n_kv, T, v_dim] TILE bf16|bfloat8_b
            indices (ttnn.Tensor): [1, n_kv, S, TOPK] uint32 block-ids (-1 = sentinel, contiguous tail).
                                   Each row must contain a valid block, and valid block ids must be < T / block_size.

        Keyword args:
            scale (float, optional): defaults to d to the power of -0.5.
            block_size (int): KV block size in tokens; defaults to 128. Must be a multiple of 32 and divide T.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional). fp8 q requires fp32_dest_acc_en.
            cache_batch_idx (int, optional): select the batch slot of a shared [B, n_kv, T, feature_dim] K/V cache.
                It is a dynamic runtime arg, so changing it (or T) does not recompile the kernels.
            chunk_start_idx (int, optional): global position of query row 0. When set, enforces a token-level
                causal mask on the diagonal block (the query's own block); toggling set/unset (None vs int)
		selects a different cached program. Requires bf16 q; fp8 q with this set is rejected.
            cluster_axis (int, optional): SP mesh axis used to derive the per-device chunk_start
                (chunk_start_idx + rank*S) under sequence parallelism. Host-side only.
            block_cyclic_sp_axis (int, optional): when set (with block_cyclic_chunk_local), the K/V cache is
                striped block-cyclic across SP on this mesh axis; the gather remaps each logical block id to its
                physical block in-kernel (invP), so no host reorder is needed. sp is read from the mesh.
            block_cyclic_chunk_local (int, optional): per-shard chunk length (chunk_size_global / sp). Required
                iff block_cyclic_sp_axis is set; cross-checked against q (must equal q_isl or tp*q_isl).

        Returns:
            ttnn.Tensor: [1, H, S, v_dim] ROW-MAJOR, dtype = q.

        Additional preconditions: d and v_dim must be multiples of 32; TOPK times 4 and output row bytes must meet
        device DRAM alignment.
        )doc",
        &ttnn::transformer::sparse_sdpa_msa,
        nb::arg("q").noconvert(),
        nb::arg("k").noconvert(),
        nb::arg("v").noconvert(),
        nb::arg("indices").noconvert(),
        nb::kw_only(),
        nb::arg("scale") = nb::none(),
        nb::arg("block_size") = 128,
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("cache_batch_idx") = nb::none(),
        nb::arg("chunk_start_idx") = nb::none(),
        nb::arg("cluster_axis") = nb::none(),
        nb::arg("block_cyclic_sp_axis") = nb::none(),
        nb::arg("block_cyclic_chunk_local") = nb::none());

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
            block_size (int, optional): Part of PagedCacheGeometryOverride (with
                `num_kv_heads`). Geometry override for an HMA-shared paged cache. When the
                K/V cache was allocated for a different layer's view, pass this call's view
                block_size (tokens/block); Q drives head_dim and the per-block element count must
                be invariant. Defaults to the cache's declared block_size.
            num_kv_heads (int, optional): Companion to `block_size` in PagedCacheGeometryOverride;
                this call's view num_kv_heads. Defaults to the cache's declared num_kv_heads.

        Returns:
            ttnn.Tensor: the output tensor [b x nqh x s x dh].

        )doc";

    ttnn::bind_function<"chunked_scaled_dot_product_attention", "ttnn.transformer.">(
        mod,
        chunked_doc,
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
        nb::arg("compute_kernel_config").noconvert() = nb::none(),
        nb::arg("block_size").noconvert() = nb::none(),
        nb::arg("num_kv_heads").noconvert() = nb::none());

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

    ttnn::bind_function<"joint_scaled_dot_product_attention", "ttnn.transformer.">(
        mod,
        joint_doc,
        &ttnn::transformer::joint_scaled_dot_product_attention,
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
        nb::arg("compute_kernel_config").noconvert() = nb::none());

    const auto* const ring_joint_doc = R"doc(
        RingJointAttention operation supports both:

        - efficient non-causal attention over two sets of query, key, and value tensors, where the first set is sharded across devices in the sequence dimension.
          Internally, these are concatenated in the sequence dimension (joint_strategy = "rear"),
          then attention is computed once. The output is split ("sliced") into two parts: one for the original Q/K/V chunk,
          and one for the joint Q/K/V chunk.
        - funtional causal attention over a single set of query, key and value tensors with the option of handling zig-zag load balancing across devices.

        This op handles optional padding via an attention mask to omit padded tokens from
        both the "original" and "joint" sequences.

        Since N must be divisible by the number of devices, the logical N must be passed in.

        Args:
            input_tensor_q (ttnn.Tensor): Original queries  [b x nh x N/num_devices x dh].
            input_tensor_k (ttnn.Tensor): Original keys     [b x nh x N/num_devices x dh].
            input_tensor_v (ttnn.Tensor): Original values [b x nhv x N/num_devices x dv].

            joint_tensor_q (ttnn.Tensor, optional): Joint queries     [b x nh x L x dh]. Defaults to None.
            joint_tensor_k (ttnn.Tensor, optional): Joint keys        [b x nh x L x dh]. Defaults to None.
            joint_tensor_v (ttnn.Tensor, optional): Joint values [b x nhv x L x dv].
                Defaults to None when L == 0.

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
            use_column_major_ccl (bool, optional): If True, allocate CCL worker cores in column-major order.
                This places CCL workers in a column (useful when reserving the last column for CCL).
                If False (default), uses row-major allocation. Defaults to False.
            is_causal (bool): Whether to use causal attention masking. Defaults to False.
            is_balanced (bool): Whether to use balanced attention computation. Defaults to False.
            is_cross (bool): Whether to use non-causal cross-attention (short Q, long K/V). Defaults to False.
            kv_cache_batch_idx (int, optional): Selects the shared K/V cache batch slot when K and V are full caches.
            kv_actual_isl (int, optional): Prior valid global KV length before this fixed-size chunk.
                When passed, enables KV-pad-aware rotation and derives current valid tokens as
                logical_n - kv_actual_isl.

        Chunked-prefill mode is entered implicitly when input_tensor_q's per-device seq
        length is less than input_tensor_k's (Q is the latest slab; K is the populated
        prefix from chunk 0 through the current chunk). The op derives chunk_size and
        the absolute Q-row offset from the shapes plus sp_size — no extra args needed.
        Chunked prefill is mathematically causal; callers must pass is_causal=True.
        When kv_cache_batch_idx is provided, input_tensor_k and input_tensor_v may be whole caches.
        The same kv_cache_batch_idx selects the K and V cache slot, and the full K/V sequence
        dimension is treated as valid. When kv_actual_isl is provided, the chunked path switches
        to KV-pad-aware rotation: logical_n remains the total valid KV length after this iteration,
        while kv_actual_isl marks the prior valid cache length before the current chunk.

        Returns:
            (ttnn.Tensor, ttnn.Tensor, ttnn.Tensor):
              - The attention output for the original Q/K/V shape [b x nh x N/num_devices x dv].
              - The attention output for the joint Q/K/V shape    [b x nh x L x dv].
              - The final log-sum-exp of the operation.           [b x nh x (N/num_devices + L) x 1]
        )doc";

    ttnn::bind_function<"ring_joint_scaled_dot_product_attention", "ttnn.transformer.">(
        mod,
        ring_joint_doc,
        &ring_joint_scaled_dot_product_attention_wrapper,
        nb::arg("input_tensor_q").noconvert(),
        nb::arg("input_tensor_k").noconvert(),
        nb::arg("input_tensor_v").noconvert(),
        nb::arg("joint_tensor_q") = nb::none(),
        nb::arg("joint_tensor_k") = nb::none(),
        nb::arg("joint_tensor_v") = nb::none(),
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
        nb::arg("ccl_core_grid_offset"),
        nb::arg("use_column_major_ccl") = false,
        nb::arg("is_causal").noconvert() = false,
        nb::arg("is_balanced").noconvert() = false,
        nb::arg("is_cross").noconvert() = false,
        nb::arg("kv_cache_batch_idx").noconvert() = nb::none(),
        nb::arg("kv_actual_isl").noconvert() = nb::none());

    const auto* const ring_mla_doc = R"doc(
        Causal Ring MLA attention over a single KV tensor.

        K and V are represented by one tensor. QK uses the full KV head dimension and
        QKT@V uses the first head_dim_v columns as V. The V dimension must be smaller
        than K and tile aligned. The KV tensor must have one shared KV head.

        Args:
            input_tensor_q (ttnn.Tensor): Queries [b x nqh x N/num_devices x dh].
            input_tensor_kv (ttnn.Tensor): Shared KV tensor [b x nkv x N/num_devices x dh].

        Keyword args:
            persistent_output_buffer_kv (ttnn.Tensor): Persistent buffer for gathered KV tensor.
            head_dim_v (int): Tile-aligned V hidden dimension, read from KV's prefix.
            logical_n (int): Logical global sequence length before sharding.
            program_config (ttnn.SDPAProgramConfig): Program configuration for the operation.
            scale (float, optional): Scale factor for QK^T. Defaults to None.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Defaults to None.
            dim (int): Dimension for ring all-gather.
            multi_device_global_semaphore (List[ttnn.GlobalSemaphore]): Global semaphores for CCL synchronization.
            num_links (int): Number of CCL links.
            cluster_axis (int): Mesh axis for all-gather.
            mesh_device (ttnn.MeshDevice): Multi-device mesh.
            topology (ttnn.ccl.Topology): Communication topology.
            subdevice_id (Optional[tt.tt_metal.SubDeviceId]): Sub-device identifier. Defaults to None.
            ccl_core_grid_offset (ttnn.CoreCoord): Core grid offset for CCL workers.
            use_column_major_ccl (bool): If true, allocate CCL workers column-major. Defaults to False.
            is_balanced (bool): Whether to use balanced causal work distribution. Defaults to False.
            kv_cache_batch_idx (int, optional): Selects one batch slot from an indexed K/V cache. Defaults to None.
            kv_actual_isl (int, optional): Prior valid global KV length before this fixed-size chunk.
                When passed, enables KV-pad-aware rotation and derives current valid tokens as
                logical_n - kv_actual_isl.

        Returns:
            (ttnn.Tensor, ttnn.Tensor):
              - Attention output [b x nqh x N/num_devices x head_dim_v].
              - Streaming statistics scratch [b x nqh x 2*N/num_devices x 1].
        )doc";

    ttnn::bind_function<"ring_mla", "ttnn.transformer.">(
        mod,
        ring_mla_doc,
        &ring_mla_wrapper,
        nb::arg("input_tensor_q").noconvert(),
        nb::arg("input_tensor_kv").noconvert(),
        nb::kw_only(),
        nb::arg("persistent_output_buffer_kv").noconvert(),
        nb::arg("head_dim_v").noconvert(),
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
        nb::arg("ccl_core_grid_offset"),
        nb::arg("use_column_major_ccl") = false,
        nb::arg("is_balanced").noconvert() = false,
        nb::arg("kv_cache_batch_idx").noconvert() = nb::none(),
        nb::arg("kv_actual_isl").noconvert() = nb::none());

    const auto* exp_ring_joint_doc = R"doc(
        ExpRingJointAttention operation that efficiently performs non-causal attention over two
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

            joint_tensor_q (ttnn.Tensor, optional): Joint queries [b x nh x L x dh]. Defaults to None (self-attention).
            joint_tensor_k (ttnn.Tensor, optional): Joint keys    [b x nh x L x dh]. Defaults to None (self-attention).
            joint_tensor_v (ttnn.Tensor, optional): Joint values  [b x nh x L x dh]. Defaults to None (self-attention).
            Pass all three joints together or omit all; an empty (zero-length) joint is treated as None.

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

        Returns:
            (ttnn.Tensor, ttnn.Tensor, ttnn.Tensor):
              - The attention output for the original Q/K/V shape [b x nh x N/num_devices x dh].
              - The attention output for the joint Q/K/V shape    [b x nh x L x dh].
              - The final log-sum-exp of the operation.           [b x nh x (N/num_devices + L) x 1]
        )doc";

    ttnn::bind_function<"exp_ring_joint_scaled_dot_product_attention", "ttnn.transformer.">(
        mod,
        exp_ring_joint_doc,
        &exp_ring_joint_scaled_dot_product_attention_wrapper,
        nb::arg("input_tensor_q").noconvert(),
        nb::arg("input_tensor_k").noconvert(),
        nb::arg("input_tensor_v").noconvert(),
        nb::arg("joint_tensor_q") = nb::none(),
        nb::arg("joint_tensor_k") = nb::none(),
        nb::arg("joint_tensor_v") = nb::none(),
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
        nb::arg("num_workers_per_link") = 1,
        nb::arg("num_buffers_per_channel") = 8);

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

    ttnn::bind_function<"flash_mla_prefill", "ttnn.transformer.">(
        mod,
        mla_doc,
        // Overload: head_dim_v as uint32_t (original MLA)
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
            nb::arg("compute_kernel_config") = nb::none()),
        // Overload: input_tensor_v as Tensor (V in embedding space)
        ttnn::overload_t(
            &flash_mla_prefill_wrapper_input_tensor,
            nb::arg("input_tensor_q").noconvert(),
            nb::arg("input_tensor_k").noconvert(),
            nb::arg("input_tensor_v").noconvert(),
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

    ttnn::bind_function<"chunked_flash_mla_prefill", "ttnn.transformer.">(
        mod,
        chunked_mla_doc,
        &ttnn::transformer::chunked_flash_mla_prefill,
        nb::arg("input_tensor_q").noconvert(),
        nb::arg("input_tensor_k").noconvert(),
        nb::arg("head_dim_v").noconvert(),
        nb::arg("page_table_tensor").noconvert(),
        nb::arg("chunk_start_idx"),
        nb::kw_only(),
        nb::arg("scale") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("program_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());

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

    ttnn::bind_function<"ring_distributed_scaled_dot_product_attention", "ttnn.transformer.">(
        mod,
        ring_distributed_doc,
        &ttnn::transformer::ring_distributed_scaled_dot_product_attention,
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
        nb::arg("chunk_start_idx") = nb::none());
}
}  // namespace ttnn::operations::transformer
