// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kv_cache.hpp"
#include "ttnn/operations/kv_cache/device/update_cache_device_operation.hpp"
#include "ttnn/operations/kv_cache/device/zero_cache_range_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn {

namespace {

// Workaround for tt-metal#42779: fill_cache / update_cache device kernels
// iterate by input_tensor.padded_shape()[-2] (not logical_shape) and write
// full tiles via the unary writer. When the input's logical seq_len is not
// tile-aligned, the implicit tile-pad rows — which may contain arbitrary
// bytes (e.g. ±Inf / NaN from upstream typecasts) — get written verbatim
// into the KV cache. SDPA later reads those positions, and although the
// causal mask sets attention[s_q < s_q_max, s_k >= S_logical] to -inf,
// 0 * Inf = NaN through softmax/V-reduce can leak into valid output lanes.
//
// Until #42779 is fixed at the kernel layer (preserve cache rows
// [S_logical, S_padded) instead of writing into them), we scrub the
// implicit tile padding of the input to zero. For tile-aligned logical
// seq_len this is a no-op via fill_implicit_tile_padding's early return.
ttnn::Tensor scrub_input_tile_padding_for_cache_write(const ttnn::Tensor& input) {
    if (input.layout() != Layout::TILE) {
        return input;
    }
    return ttnn::fill_implicit_tile_padding(input, 0.0f, input.memory_config());
}

}  // namespace

ttnn::Tensor update_cache_for_token_(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    const uint32_t update_index,
    const uint32_t batch_offset,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    auto kernel_config_val = init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config);
    auto scrubbed_input = scrub_input_tile_padding_for_cache_write(input);
    ttnn::prim::update_cache(
        cache, scrubbed_input, 0, update_index, batch_offset, ttnn::prim::UpdateCacheOpType::UPDATE, kernel_config_val);
    return cache;
}

ttnn::Tensor fill_cache_for_user_(
    const ttnn::Tensor& cache, const ttnn::Tensor& input, const uint32_t batch_index) {
    auto scrubbed_input = scrub_input_tile_padding_for_cache_write(input);
    ttnn::prim::update_cache(cache, scrubbed_input, batch_index, 0, 0, ttnn::prim::UpdateCacheOpType::FILL);
    return cache;
}

ttnn::Tensor update_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    const uint32_t update_idx,
    const uint32_t batch_offset,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    auto kernel_config_val = init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config);
    auto scrubbed_input = scrub_input_tile_padding_for_cache_write(input);
    ttnn::prim::update_cache(
        cache, scrubbed_input, 0, update_idx, batch_offset, ttnn::prim::UpdateCacheOpType::UPDATE, kernel_config_val);
    return cache;
}

ttnn::Tensor fill_cache(
    const ttnn::Tensor& cache_tensor, const ttnn::Tensor& input_tensor, const uint32_t batch_idx) {
    auto scrubbed_input = scrub_input_tile_padding_for_cache_write(input_tensor);
    ttnn::prim::update_cache(cache_tensor, scrubbed_input, batch_idx, 0, 0, ttnn::prim::UpdateCacheOpType::FILL);
    return cache_tensor;
}

ttnn::Tensor zero_cache_range(const ttnn::Tensor& cache, const uint32_t start_token, const uint32_t end_token) {
    using namespace tt::constants;
    uint32_t Wt = cache.padded_shape()[-1] / TILE_WIDTH;
    // Round start_token down to tile boundary, end_token up to tile boundary
    uint32_t start_page = (start_token / TILE_HEIGHT) * Wt;
    uint32_t end_page = ((end_token + TILE_HEIGHT - 1) / TILE_HEIGHT) * Wt;
    ttnn::prim::zero_cache_range(cache, start_page, end_page);
    return cache;
}

}  // namespace ttnn
