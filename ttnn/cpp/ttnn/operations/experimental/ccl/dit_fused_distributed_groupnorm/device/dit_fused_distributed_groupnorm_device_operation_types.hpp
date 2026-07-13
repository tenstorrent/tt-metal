// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt_stl/reflection.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

// Mirrors GroupNormParams, plus CCL fields for stats all-gather on cluster_axis.
struct DitFusedDistributedGroupnormParams {
    float eps = 0.0f;
    uint32_t num_groups = 0;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
    bool use_welford = false;

    uint32_t cluster_axis = 0;
    uint32_t num_links = 1;
    uint32_t ring_size = 1;
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear;
    std::vector<GlobalSemaphore> multi_device_global_semaphore;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;

    auto attributes() const {
        using ttsl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("eps", eps);
        attrs.emplace_back("num_groups", num_groups);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("compute_kernel_config", compute_kernel_config);
        attrs.emplace_back("use_welford", use_welford);
        attrs.emplace_back("cluster_axis", cluster_axis);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("topology", topology);
        return attrs;
    }
};

// Mirrors GroupNormInputs (gamma/beta), plus persistent stats buffer for AG.
struct DitFusedDistributedGroupnormInputs {
    Tensor input;
    std::optional<Tensor> gamma;
    std::optional<Tensor> beta;
    std::optional<Tensor> input_mask;
    std::optional<Tensor> persistent_output_buffer;
};

// Multi-core mcast GroupNorm layout: cores split into num_virtual_cols group-columns × spatial-row
// stacks (exactly like the stock groupnorm_mcast layout). One mcast-group master per group-column
// owns num_groups/num_virtual_cols groups; masters = num_virtual_cols (for N==1). This helper is the
// single source of truth for the master count so create_stats_buffer and the program factory agree.
uint32_t gn_num_virtual_cols(uint32_t channels, uint32_t num_groups, uint32_t grid_x);

// Stats stick: PER MASTER, bf16 [mean, var] over that master's num_groups/num_virtual_cols groups.
// stick_bytes = round_up(num_groups_per_core * 4, 64) — a multiple of NOC_DRAM_READ_ALIGNMENT_BYTES
// (64 on Blackhole) because each master NoC-reads its own sub-stick at DRAM offset slot*stick_bytes,
// and non-64-aligned offsets read back as zero on BH. A single forwarder coalesces all masters'
// sub-sticks into one packet (num_groups*4 B ≤ one fabric packet), so num_chunks_per_device = 1,
// total_pages = ring_size, and the DRAM page = num_masters * stick_bytes.
struct DitFusedDistributedGroupnormSizing {
    bool is_local = false;
    uint32_t num_groups = 0;
    uint32_t num_masters = 1;            // mcast-group senders per device = num_virtual_cols
    uint32_t num_groups_per_core = 0;    // groups owned by each master
    uint32_t num_forwarders = 0;         // single coalescing forwarder (optimal for the tiny stick)
    uint32_t num_chunks_per_device = 0;  // = num_forwarders (max_rounds == 1)
    uint32_t stick_bytes = 0;            // per-master, 64 B-aligned
    uint32_t total_pages = 0;            // = ring_size
    uint32_t page_size_bytes = 0;        // = num_masters * stick_bytes
};

// Single source of truth for the stats-buffer geometry. Used by create_stats_buffer (host
// allocation), compute_output_specs, validate, and the program factory so they cannot drift.
DitFusedDistributedGroupnormSizing gn_make_sizing(
    uint32_t num_groups, uint32_t ring_size, uint32_t channels, uint32_t grid_x);

DitFusedDistributedGroupnormSizing compute_sizing(const DitFusedDistributedGroupnormParams& args, const Tensor& input);

}  // namespace ttnn::experimental::prim
