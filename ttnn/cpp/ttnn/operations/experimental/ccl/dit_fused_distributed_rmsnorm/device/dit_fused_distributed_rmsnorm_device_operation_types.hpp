// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/core_coord.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/ccl/dit_fused_distributed_rmsnorm/dit_fused_distributed_rmsnorm.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace ttnn::experimental::prim {

using ttnn::experimental::DitFusedNormType;

// Attributes for the fused Wan2.2 distributed RMSNorm device op.
// Combines: per-row RMSNorm pre stats, ring all-gather of stats across the TP
// cluster axis, post normalization, optional head-split, optional RoPE, and
// optional output-dtype cast — all in a single program with L1-resident input.
struct DitFusedDistributedRmsnormParams {
    float epsilon;
    uint32_t num_heads_per_device;
    // Per-head normalization (FLUX.2 path): reduce over head_dim per
    // (token, head) instead of the full row. When true, AG is skipped
    // entirely — each head is assumed local to chip.
    bool per_head_norm;

    // Selects RMSNorm (sum-of-squares) vs Welford LayerNorm (mean/variance).
    // Defaults to RMS so all existing call sites are unchanged.
    DitFusedNormType norm_type;

    // Output dtype override (defaults to input dtype if unset).
    std::optional<DataType> dtype;
    MemoryConfig output_mem_config;

    // CCL config
    uint32_t cluster_axis;
    uint32_t num_links;
    uint32_t ring_size;
    ttnn::ccl::Topology topology;
    std::vector<GlobalSemaphore> multi_device_global_semaphore;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;

    DeviceComputeKernelConfig compute_kernel_config;

    DitFusedDistributedRmsnormParams(
        float epsilon,
        uint32_t num_heads_per_device,
        bool per_head_norm,
        std::optional<DataType> dtype,
        MemoryConfig output_mem_config,
        uint32_t cluster_axis,
        uint32_t num_links,
        uint32_t ring_size,
        ttnn::ccl::Topology topology,
        std::vector<GlobalSemaphore> multi_device_global_semaphore,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
        DeviceComputeKernelConfig compute_kernel_config,
        DitFusedNormType norm_type = DitFusedNormType::RMS) :
        epsilon(epsilon),
        num_heads_per_device(num_heads_per_device),
        per_head_norm(per_head_norm),
        norm_type(norm_type),
        dtype(dtype),
        output_mem_config(std::move(output_mem_config)),
        cluster_axis(cluster_axis),
        num_links(num_links),
        ring_size(ring_size),
        topology(topology),
        multi_device_global_semaphore(std::move(multi_device_global_semaphore)),
        sub_device_id(sub_device_id),
        compute_kernel_config(compute_kernel_config) {}

    auto attributes() const {
        using ttsl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("epsilon", epsilon);
        attrs.emplace_back("num_heads_per_device", num_heads_per_device);
        attrs.emplace_back("per_head_norm", per_head_norm);
        attrs.emplace_back("norm_type", static_cast<uint8_t>(norm_type));
        attrs.emplace_back("dtype", dtype);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("cluster_axis", cluster_axis);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("compute_kernel_config", compute_kernel_config);
        return attrs;
    }
};

struct DitFusedDistributedRmsnormInputs {
    Tensor input;
    std::optional<const Tensor> weight;
    std::optional<const Tensor> bias;
    std::optional<const Tensor> transformation_mat;
    std::optional<const Tensor> rope_cos;
    std::optional<const Tensor> rope_sin;
    // Persistent buffer for the AG of stats (optional; allocated internally if null).
    std::optional<Tensor> persistent_output_buffer;
    // Welford LayerNorm reciprocal LUT: a row-major fp32 [1, reduce_width] DRAM tensor
    // holding [1/1, 1/2, ..., 1/reduce_width] (== ttnn.create_layer_norm_reciprocals),
    // replicated per device. The writer NoC-reads it into a CB at kernel start so the
    // Welford LLK does an array load instead of a soft-float 1/(N+1) per sample. Only
    // consumed on the LAYERNORM path; ignored (may be null) for RMS.
    std::optional<const Tensor> reciprocals;
};

// Sizing derived from args + input shape. Shared between compute_output_specs
// (to spec the stats DRAM scratch tensor) and the program factory (to lay out
// kernels and CBs identically). Keep all derivation in one place so the spec
// and the factory cannot drift.
//
// Packed-page stats all-gather: the post-reduce stat tile carries only
// 32 real values (one per token in the tile-row, in col 0). Transposing the
// tile moves those 32 values to row 0 — split across face_00[0..63] and
// face_01[0..63]. We then pack `window_size` such row-0 strips into one
// row-major page of `TILE_HEIGHT * window_size * elem_bytes` bytes and
// fabric-mcast the page in a single packet. That cuts both fabric and DRAM
// traffic by ~32× compared to sending whole 4 KB tiles.
//
// The buffer's logical shape is independent of `num_workers` per design
// constraint — workers cooperate by writing different chunk indices into
// the same set of pages. Total pages per chip = num_devices * num_chunks_per_device.
struct DitFusedDistributedRmsnormSizing {
    uint32_t num_tile_rows = 0;
    uint32_t num_workers = 0;
    bool is_tp_1 = false;                // ring_size==1 or per_head_norm: reduce locally, no all-gather
    bool use_mux = false;                // !is_tp_1: uses the fabric-forwarder all-gather + DRAM scratch
    uint32_t window_size = 0;            // tile-rows per packed page
    uint32_t num_chunks_per_device = 0;  // ceil(num_tile_rows / window_size)
    uint32_t total_pages = 0;            // num_devices * num_chunks_per_device (0 on the is_tp_1 path)
    uint32_t page_size_bytes = 0;        // TILE_HEIGHT * window_size * sizeof(float)
    // Stats transported per token-tile: 1 for RMSNorm (sum-of-squares), 2 for
    // Welford LayerNorm (mean, M2). Each stat is a 128 B packed stick, so the
    // physical stick is stats_per_token * 128 B.
    uint32_t stats_per_token = 1;
    uint32_t stick_bytes = 128;  // stats_per_token * 128
};

DitFusedDistributedRmsnormSizing compute_sizing(
    const DitFusedDistributedRmsnormParams& args, const Tensor& input);

// Single source of truth for the persistent gathered-stats DRAM scratch spec:
// [1, 1, total_pages, TILE_HEIGHT * window_size], FLOAT32, ROW_MAJOR, DRAM INTERLEAVED.
// Used by the pre-alloc helper, compute_output_specs, and validate so the pre-allocated
// buffer, the op's expected spec, and the validation check cannot drift. Only meaningful
// on the all-gather path (sizing.use_mux; total_pages > 0).
tt::tt_metal::TensorSpec make_stats_tensor_spec(const DitFusedDistributedRmsnormSizing& sizing);

}  // namespace ttnn::experimental::prim
