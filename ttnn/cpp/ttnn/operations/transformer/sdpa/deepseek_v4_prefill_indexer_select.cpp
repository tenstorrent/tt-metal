// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/deepseek_v4_prefill_indexer_select.hpp"
#include "ttnn/operations/transformer/sdpa/device/deepseek_v4_prefill_indexer_select_device_operation.hpp"
#include <tt-metalium/hal.hpp>

namespace ttnn::transformer {

namespace {
// fp32 DEST and full sync are forced by the op (TopK XL); only fidelity and approx mode are taken from here.
ttnn::DeviceComputeKernelConfig kernel_config(std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    return init_device_compute_kernel_config(
        tt::tt_metal::hal::get_arch(),
        compute_kernel_config,
        /*default_fidelity=*/MathFidelity::HiFi2,
        /*default_approx_mode=*/true,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
}
}  // namespace

ttnn::Tensor deepseek_v4_prefill_indexer_select(
    const ttnn::Tensor& q,
    const ttnn::Tensor& weights,
    const ttnn::Tensor& idx_keys,
    const ttnn::Tensor& start,
    uint32_t compress_ratio,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    return ttnn::prim::deepseek_v4_prefill_indexer_select(
        q, weights, idx_keys, start, std::nullopt, compress_ratio, false, kernel_config(compute_kernel_config));
}

ttnn::Tensor deepseek_v4_prefill_indexer_select_candidates(
    const ttnn::Tensor& q,
    const ttnn::Tensor& weights,
    const ttnn::Tensor& idx_keys,
    const ttnn::Tensor& start,
    uint32_t compress_ratio,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    return ttnn::prim::deepseek_v4_prefill_indexer_select(
        q, weights, idx_keys, start, std::nullopt, compress_ratio, true, kernel_config(compute_kernel_config));
}

ttnn::Tensor deepseek_v4_prefill_indexer_select_from_candidates(
    const ttnn::Tensor& q,
    const ttnn::Tensor& weights,
    const ttnn::Tensor& idx_keys,
    const ttnn::Tensor& start,
    const ttnn::Tensor& candidates,
    uint32_t compress_ratio,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    return ttnn::prim::deepseek_v4_prefill_indexer_select(
        q, weights, idx_keys, start, candidates, compress_ratio, false, kernel_config(compute_kernel_config));
}

}  // namespace ttnn::transformer
