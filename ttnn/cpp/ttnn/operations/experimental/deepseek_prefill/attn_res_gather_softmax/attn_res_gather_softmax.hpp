// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::attn_res_gather_softmax {

// One read site's whole path from a tensor-parallel-sharded residual stream to the mixed
// hidden state, in a single dispatch. The path is three stages:
//
//   stats  per-rank sum of squares and dots over this rank's shard of `d`
//   gather completes them across the shard axis
//   fold   the online-softmax combine below
//
// and the fold itself is
//
//   live_scores = sum_p dots_p * rsqrt(sum_p sum_squares_p * inv_hidden_size + eps)
//   m   = max(shift, live_scores)
//   h   = (partial * exp(shift - m) + running_sum * exp(live_scores - m))
//         / (mass * exp(shift - m) + exp(live_scores - m))
//
// The wire carries the statistics, not the reduction: what crosses is two scalars per
// token either way, and summing them on arrival is a pair of dest-register adds inside a
// pass that already holds those tiles, while reducing on the wire would cost a second
// device program.
//
// `stats` is caller-allocated scratch of shape `[1, 2 * ring_size, N, 1]` in the same
// dtype as `shift` and `mass`. It is not read on entry and holds nothing meaningful on
// exit; it exists as an operand so a caller walking many read sites allocates it once.
// It must be allocated across the whole mesh — the exchange addresses a peer's slot by
// page of that peer's own copy.
//
// `partial`, `shift` and `mass` may each carry R read sites on dim 0, with `site` picking
// the plane; at R == 1 an operand is shared by every site and `site` does not apply to
// it. `site` shapes no kernel, so walking R sites reuses one cached program.
//
// `pending` is a residual write the caller has not put into the stream yet. Given, the op
// scores and folds `running_sum + pending` and returns that sum as a second output for the
// caller to carry forward, which is a whole dispatch cheaper than adding it beforehand.
// The returned vector is one tensor without it and two with it.
std::vector<ttnn::Tensor> attn_res_gather_softmax(
    const ttnn::Tensor& partial,
    const ttnn::Tensor& running_sum,
    const ttnn::Tensor& shift,
    const ttnn::Tensor& mass,
    const ttnn::Tensor& q,
    const ttnn::Tensor& stats,
    const GlobalSemaphore& semaphore,
    uint32_t cluster_axis,
    uint32_t site,
    float inv_hidden_size,
    float eps,
    const std::optional<ttnn::Tensor>& pending,
    std::optional<uint32_t> num_links,
    std::optional<ttnn::ccl::Topology> topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config);

}  // namespace ttnn::operations::experimental::deepseek_prefill::attn_res_gather_softmax
