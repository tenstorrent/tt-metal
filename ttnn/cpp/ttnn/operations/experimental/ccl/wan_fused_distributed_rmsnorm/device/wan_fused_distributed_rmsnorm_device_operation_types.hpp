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
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// Attributes for the fused Wan2.2 distributed RMSNorm device op.
// Combines: per-row RMSNorm pre stats, ring all-gather of stats across the TP
// cluster axis, post normalization, optional head-split, optional RoPE, and
// optional output-dtype cast — all in a single program with L1-resident input.
struct WanFusedDistributedRmsnormParams {
    float epsilon;
    uint32_t num_heads_per_device;
    // Per-head normalization (FLUX.2 path): reduce over head_dim per
    // (token, head) instead of the full row. When true, AG is skipped
    // entirely — each head is assumed local to chip.
    bool per_head_norm;

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

    WanFusedDistributedRmsnormParams(
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
        DeviceComputeKernelConfig compute_kernel_config) :
        epsilon(epsilon),
        num_heads_per_device(num_heads_per_device),
        per_head_norm(per_head_norm),
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

struct WanFusedDistributedRmsnormInputs {
    Tensor input;
    std::optional<const Tensor> weight;
    std::optional<const Tensor> bias;
    std::optional<const Tensor> transformation_mat;
    std::optional<const Tensor> rope_cos;
    std::optional<const Tensor> rope_sin;
    // Persistent buffer for the AG of stats (optional; allocated internally if null).
    std::optional<Tensor> persistent_output_buffer;
};

// Sizing derived from args + input shape. Shared between compute_output_specs
// (to spec the stats DRAM scratch tensor) and the program factory (to lay out
// kernels and CBs identically). Keep all derivation in one place so the spec
// and the factory cannot drift.
//
// Packed-page stats AG (Phase 9): the post-reduce stat tile carries only
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
struct WanFusedDistributedRmsnormSizing {
    uint32_t num_tile_rows = 0;
    uint32_t num_workers = 0;
    bool is_tp_1 = false;
    bool use_mux = false;
    uint32_t chunk_size_rows = 0;        // same as window_size; kept for legacy field names
    uint32_t window_size = 0;            // tile-rows per packed page
    uint32_t num_chunks_per_device = 0;  // ceil(num_tile_rows / window_size)
    uint32_t total_pages = 0;            // num_devices * num_chunks_per_device (0 when !use_mux)
    uint32_t page_size_bytes = 0;        // TILE_HEIGHT * window_size * sizeof(float)
};

WanFusedDistributedRmsnormSizing compute_sizing(const WanFusedDistributedRmsnormParams& args, const Tensor& input);

}  // namespace ttnn::experimental::prim
