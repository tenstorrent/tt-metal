// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/sub_device_types.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct AttnResGatherSoftmaxParams {
    uint32_t site;
    float inv_hidden_size;
    float eps;
    tt::tt_metal::MemoryConfig output_mem_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
    ttnn::ccl::Topology topology;
    uint32_t num_links;
    // Tensor-parallel ranks over `cluster_axis`, and so also the number of statistic
    // pairs the fold sums.
    uint32_t ring_size;
    uint32_t cluster_axis;
    tt::tt_metal::GlobalSemaphore semaphore;  // Not default constructible
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;

    AttnResGatherSoftmaxParams(
        uint32_t site,
        float inv_hidden_size,
        float eps,
        tt::tt_metal::MemoryConfig output_mem_config,
        ttnn::DeviceComputeKernelConfig compute_kernel_config,
        ttnn::ccl::Topology topology,
        uint32_t num_links,
        uint32_t ring_size,
        uint32_t cluster_axis,
        tt::tt_metal::GlobalSemaphore semaphore,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id) :
        site(site),
        inv_hidden_size(inv_hidden_size),
        eps(eps),
        output_mem_config(std::move(output_mem_config)),
        compute_kernel_config(compute_kernel_config),
        topology(topology),
        num_links(num_links),
        ring_size(ring_size),
        cluster_axis(cluster_axis),
        semaphore(std::move(semaphore)),
        sub_device_id(sub_device_id) {}

    // Two exclusions, both so that one cached program serves a whole walk. `site`
    // resolves to page offsets the dataflow kernels take as common runtime args, and
    // the semaphore resolves to an address the gather kernel takes as a runtime arg;
    // both are rewritten per dispatch. Without that, a caller walking R read sites
    // compiles and caches R copies of one program, which is the cost this op exists to
    // avoid.
    auto attributes() const {
        using ttsl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.reserve(8);
        attrs.emplace_back("inv_hidden_size", inv_hidden_size);
        attrs.emplace_back("eps", eps);
        attrs.emplace_back("output_mem_config", output_mem_config);
        attrs.emplace_back("compute_kernel_config", compute_kernel_config);
        attrs.emplace_back("topology", topology);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("ring_size", ring_size);
        attrs.emplace_back("cluster_axis", cluster_axis);
        return attrs;
    }
};

// One read site's whole path from the live stream to the mixed hidden state: this
// rank's share of the scoring statistics, the exchange that completes them across the
// tensor-parallel axis, and the online-softmax fold. Unfused that is three dispatches.
//
// partial - [R, 1, H, W], TILE layout: the sealed snapshots' mixture, one plane per
//   read site, passed whole. `site` picks the plane.
// running_sum - [1, 1, H, W]: the live residual stream. Both the tensor that gets
//   scored and the fold's second operand. Its dtype must match `partial`'s — the two
//   share one unpacker configuration in the fold.
// pending - [1, 1, H, W], optional: a residual write into the stream that has not been
//   settled yet. Given, the op scores and folds `running_sum + pending` instead of
//   `running_sum` and returns the sum as a second output, which is what the caller's
//   stream carries forward. The sum is formed in the pass that already reads
//   `running_sum` tile by tile, so folding it in costs a read and a write rather than a
//   dispatch.
// shift, mass - [R, 1, H, 1], TILE layout. One scalar per row, physically a tile whose
//   column 0 holds it, which is already what BroadcastType::COL reads.
// q - [1, 1, 1, W]: the folded query the statistics are taken against.
// stats - [1, 2 * ring_size, H, 1], TILE layout, and the same dtype as shift and mass:
//   scratch, not read on entry and not meaningful on exit. Rank p writes its sum of
//   squares to plane 2p and its dots to plane 2p+1, so a peer's slot is addressed by
//   rank alone. It must be device-allocated across the whole mesh — the exchange names
//   a peer's page by its local address — and it is an operand rather than an internal
//   allocation so that a caller walking 186 read sites allocates it once.
//
// At R == 1 an operand is shared by every site and `site` does not apply to it, which
// is what lets one call mix batched partial, shift and mass with a per-site stream.
struct AttnResGatherSoftmaxInputs {
    ttnn::Tensor partial;
    ttnn::Tensor running_sum;
    ttnn::Tensor shift;
    ttnn::Tensor mass;
    ttnn::Tensor q;
    ttnn::Tensor stats;
    std::optional<ttnn::Tensor> pending;
};

}  // namespace ttnn::experimental::prim
