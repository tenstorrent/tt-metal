// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rjsdpa_nanobind.hpp"
#include "rjsdpa.hpp"
#include "update_cache.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

namespace ttnn::experimental::gemma4 {
namespace nb = nanobind;
using operations::transformer::SDPAProgramConfig;

namespace {
std::tuple<ttnn::Tensor, ttnn::Tensor> rjsdpa_wrapper(
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
    std::optional<uint32_t> cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    CoreCoord ccl_core_grid_offset,
    bool use_column_major_ccl,
    bool is_balanced,
    std::optional<uint32_t> kv_cache_batch_idx,
    std::optional<uint32_t> kv_actual_isl,
    const std::optional<ttnn::Tensor>& slot_id,
    const std::optional<ttnn::Tensor>& kv_actual_isl_tensor,
    std::optional<uint32_t> kv_cache_num_layers,
    std::optional<uint32_t> kv_cache_layer_idx,
    std::optional<uint32_t> stripe_size,
    const std::optional<ttnn::Tensor>& input_tensor_v,
    const std::optional<ttnn::Tensor>& persistent_output_buffer_v,
    std::optional<uint32_t> sliding_window_size,
    const std::string& communication,
    const std::optional<ttnn::Tensor>& request_descriptors,
    std::optional<uint32_t> request_index,
    std::optional<uint32_t> request_local_tokens,
    const std::optional<ttnn::Tensor>& output_tensor,
    const std::optional<ttnn::Tensor>& stats_tensor) {
    auto strategy = use_column_major_ccl ? ttnn::ccl::CoreAllocationStrategy::COL_MAJOR
                                         : ttnn::ccl::CoreAllocationStrategy::ROW_MAJOR;
    return ttnn::experimental::gemma4::rjsdpa(
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
        kv_actual_isl,
        slot_id,
        kv_actual_isl_tensor,
        kv_cache_num_layers,
        kv_cache_layer_idx,
        stripe_size,
        input_tensor_v,
        persistent_output_buffer_v,
        sliding_window_size,
        communication,
        request_descriptors,
        request_index,
        request_local_tokens,
        output_tensor,
        stats_tensor);
}
}  // namespace

void bind_rjsdpa(nb::module_& mod) {
    auto gemma4 = mod.def_submodule("gemma4", "Experimental Gemma4 operations");
    ttnn::bind_function<"update_cache", "ttnn.experimental.gemma4.">(
        gemma4,
        "Copy a descriptor's rows from a packed TILE buffer into a stripe=32 KV cache. "
        "request_local_tokens is the fixed per-rank capacity for this trace; the descriptor "
        "selects the source offset, cache slot, prefix, and valid token count.",
        &update_cache,
        nb::arg("cache").noconvert(),
        nb::arg("input").noconvert(),
        nb::kw_only(),
        nb::arg("request_descriptors").noconvert(),
        nb::arg("request_index"),
        nb::arg("request_local_tokens"),
        nb::arg("layer_idx"),
        nb::arg("num_layers"),
        nb::arg("cluster_axis"));
    const auto* doc = R"doc(
        Packed-KV causal ring attention, with the same arguments and (output, statistics)
        return contract as ttnn.transformer.ring_mla, plus optional stripe_size.
        Separate K/V tensors and causal sliding-window attention are also supported.

        With stripe_size=None, behavior is identical to ring_mla (including its supported
        latent-V, packed-KV, balanced, indexed-cache and device-metadata modes).
        For packed KV, QK reads the first Q.head_dim columns, and V the last head_dim_v
        columns. For an unexpanded latent KV tensor, V is its first head_dim_v columns.

        stripe_size (int, optional): Ownership stripe in TOKENS PER CP RANK, independent
        of the total local Q length and program_config.q_chunk_size. A cache row j on
        rank r represents global position
            (j // stripe_size) * (CP * stripe_size) + r * stripe_size + j % stripe_size.
        Q packs this rank's rows from [prefix, logical_n), in increasing global position.
        For CP=8, stripe_size=32 and 64 local Q rows, rank 0 holds [0,32), [256,288).
        All query stripes are processed in ONE fused ring-attention dispatch.

        Explicit stripes require unbalanced causal attention with matching Q/KV sequence
        distributions, whole-tile prefixes, and local Q/KV lengths divisible by stripe_size.
        The prefix is kv_actual_isl when provided; otherwise it is inferred as
        max(0, logical_n - Q.local_seq * CP). Host lengths must be tile-aligned.
        slot_id and kv_actual_isl_tensor retain ring_mla's trace-safe contract: the valid
        end is min(device_prefix + Q.local_seq * CP, cache_capacity), regardless of stripe_size.
        Select cache users/layers exactly as in ring_mla; this is a single-request stripe
        extension, not a ragged multi-request interface. The cache must already use the
        specified stripe layout. FP32 destination accumulation and split-KV are unsupported
        with explicit stripes. Omitting stripe_size retains all legacy modes.

        For sliding attention, input_tensor_kv is K; provide input_tensor_v and its
        persistent_output_buffer_v. head_dim_v must match V's width. Set
        sliding_window_size=W to attend exactly max(0, q-W+1) <= k <= q.
        With explicit stripes the op uses one fused dispatch and gathers only a bounded
        tail covering the window and current queries. Persistent buffers retain the full
        rank-major gathered-cache shape (CP * local cache rows); they are not compact halo
        buffers. Communication and compute do not grow with the prefix length.

        With explicit sliding stripes, communication="all_gather" (default) uses the bounded
        gathered tail. communication="halo" instead sends each destination's needed KV
        interval directly, multicasting identical intervals along each fabric direction.
        Halo currently requires a CP axis and Linear 1D fabric; it retains the same full
        gathered scratch shape, token ownership, math, metadata, and output contract.

        At CP=8, stripe_size=32 supports 256-token chunks and every multiple of 256.
        q_chunk_size must divide the local Q length; k_chunk_size must divide the local
        cache length. Both are positive multiples of 32, independently of stripe_size.
        Sliding requires separate BF16 Q / BFP8 K,V with equal Q/K/V head dimensions and
        matching K/V head counts. Host or device prefix, indexed users/layers, partial
        queries, and trace replay are supported. Circular caches are not supported with
        explicit stripes. Without stripe_size, sliding retains the legacy halo contract.

        Packed-buffer addressing (stripe_size=32): supply request_descriptors, request_index,
        request_local_tokens, and output_tensor together. Descriptors are replicated UINT32
        ROW_MAJOR DRAM [max_requests, 8] rows:
            [active, cache_slot, prefix_tokens, valid_tokens,
             packed_offset_tiles, packed_span_tiles, 0, 0].
        Offsets/spans count 32-token rows per CP rank. request_local_tokens is a fixed,
        compiled request capacity and must equal packed_span_tiles * 32. The selected
        descriptor must be active; inactive dispatches and dynamic capacities are not supported.
        Descriptors replace slot_id / kv_actual_isl_tensor. Q and output_tensor contain the
        full packed batch; only this request's span is read/written, with their full head
        strides. output_tensor must be BF16 TILE DRAM with Q's shape and head_dim_v width.
        Other output rows are untouched; initialize them before each replay if needed.
        valid_tokens bounds KV updates; SDPA retains padded-capacity causal scheduling, and
        callers ignore padded query results.

        Optional stats_tensor reuses BF16 TILE DRAM scratch of shape
        [1, Q.heads, >=2*request_local_tokens, 1], with tile-aligned sequence length.
        Request calls may share it sequentially. Preallocated output/statistics are returned
        by reference; buffers must not alias inputs or each other.
        )doc";
    ttnn::bind_function<"rjsdpa", "ttnn.experimental.gemma4.">(
        gemma4,
        doc,
        &rjsdpa_wrapper,
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
        nb::arg("kv_actual_isl").noconvert() = nb::none(),
        nb::arg("slot_id").noconvert() = nb::none(),
        nb::arg("kv_actual_isl_tensor").noconvert() = nb::none(),
        nb::arg("kv_cache_num_layers").noconvert() = nb::none(),
        nb::arg("kv_cache_layer_idx").noconvert() = nb::none(),
        nb::arg("stripe_size").noconvert() = nb::none(),
        nb::arg("input_tensor_v").noconvert() = nb::none(),
        nb::arg("persistent_output_buffer_v").noconvert() = nb::none(),
        nb::arg("sliding_window_size").noconvert() = nb::none(),
        nb::arg("communication") = "all_gather",
        nb::arg("request_descriptors").noconvert() = nb::none(),
        nb::arg("request_index") = nb::none(),
        nb::arg("request_local_tokens") = nb::none(),
        nb::arg("output_tensor").noconvert() = nb::none(),
        nb::arg("stats_tensor").noconvert() = nb::none());
}

}  // namespace ttnn::experimental::gemma4
