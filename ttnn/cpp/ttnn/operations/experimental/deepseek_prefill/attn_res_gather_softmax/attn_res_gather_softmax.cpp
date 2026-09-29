// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attn_res_gather_softmax.hpp"

#include "device/attn_res_gather_softmax_device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::attn_res_gather_softmax {

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
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    TT_FATAL(
        partial.storage_type() == StorageType::DEVICE,
        "Input tensor storage type must be DEVICE but got {}",
        partial.storage_type());

    // HiFi4 with fp32 dest accumulation. The statistics reduce and the whole weight
    // derivation run in dst, and `exp` then `recip` on a bf16 dest would round the
    // denominator twice before it ever divides a value.
    auto kernel_config_val = init_device_compute_kernel_config(
        partial.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true);

    // Downgrades a requested Ring to Linear when the mesh cannot close the axis, which
    // is what decides whether a rank has a backward neighbour at index 0.
    const auto usable_topology = ttnn::ccl::get_usable_topology(partial, topology, cluster_axis);

    return ttnn::prim::attn_res_gather_softmax(
        partial,
        running_sum,
        shift,
        mass,
        q,
        stats,
        pending,
        site,
        inv_hidden_size,
        eps,
        cluster_axis,
        *partial.device(),
        semaphore,
        usable_topology,
        num_links.value_or(1),
        subdevice_id,
        memory_config.value_or(partial.memory_config()),
        kernel_config_val);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::attn_res_gather_softmax
