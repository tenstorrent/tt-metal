// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include <optional>
#include <tuple>

namespace ttnn::transformer {

// Sol-Attn ring attention, forked from exp_ring_joint_scaled_dot_product_attention.
//
// Currently an exact clone of its parent: the fork exists so the block-sparse routing and the
// pooled-summary residual can be added against a known-good baseline rather than grown inside the
// dense op. It is registered separately so the dense path keeps its program cache and its
// measured behaviour while this one changes.
//
// The reason for forking the *ring* kernel rather than vsa_sdpa: at full density vsa_sdpa spends
// 103.7 ms on the attention the dense ring op does in 51.4 ms, so its streaming design carries a
// ~2x structural penalty that a block-sparse consumer would inherit. The ring also delivers each
// shard's K/V contiguously as it rotates, which is what a scattered DRAM gather pays ~9.6 ms of
// k-independent cost to emulate.
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> sol_ring(
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
    operations::transformer::SDPAProgramConfig program_config,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<float> scale,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    uint32_t num_workers_per_link,
    uint32_t num_buffers_per_channel,
    // packed-bit [H, n_q_chunks, n_kv_chunks] chunk-skip mask; absent = attend every chunk
    std::optional<ttnn::Tensor> chunk_mask = std::nullopt,
    // experiment knob: keep every Nth KV chunk, skip the rest. 0 = dense.
    uint32_t chunk_keep_every = 0);

}  // namespace ttnn::transformer
