// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/deepseek_v4_prefill_sdpa_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <bit>

namespace ttnn::prim {

namespace {
void check_row_major_dram(const Tensor& t, const char* name) {
    TT_FATAL(
        t.storage_type() == StorageType::DEVICE && t.buffer() != nullptr,
        "deepseek_v4_prefill_sdpa: {} must be on device",
        name);
    TT_FATAL(t.layout() == Layout::ROW_MAJOR, "deepseek_v4_prefill_sdpa: {} must be ROW_MAJOR", name);
    TT_FATAL(t.memory_config().buffer_type() == BufferType::DRAM, "deepseek_v4_prefill_sdpa: {} must be in DRAM", name);
    TT_FATAL(!t.memory_config().is_sharded(), "deepseek_v4_prefill_sdpa: {} must be interleaved", name);
    TT_FATAL(t.padded_shape() == t.logical_shape(), "deepseek_v4_prefill_sdpa: {} must not be padded", name);
}

// What the program hash does not key on, checked on every dispatch: placement, layout, and the cache length T
// (which rides on runtime args).
void validate_non_hashed(const DeepseekV4PrefillSDPAInputs& t) {
    const auto* device = t.q.device();
    check_row_major_dram(t.q, "q");
    check_row_major_dram(t.window_kv, "window_kv");
    check_row_major_dram(t.start, "start");
    TT_FATAL(t.window_kv.device() == device, "deepseek_v4_prefill_sdpa: window_kv must be on q's device");
    TT_FATAL(t.start.device() == device, "deepseek_v4_prefill_sdpa: start must be on q's device");
    TT_FATAL(t.start.logical_volume() == 1, "deepseek_v4_prefill_sdpa: start must hold exactly one element");
    const uint32_t K_DIM = t.q.logical_shape()[3];
    if (t.cache_kv.has_value()) {
        const auto& cache = t.cache_kv.value();
        check_row_major_dram(cache, "cache_kv");
        check_row_major_dram(t.cache_idx.value(), "cache_idx");
        TT_FATAL(cache.device() == device, "deepseek_v4_prefill_sdpa: cache_kv must be on q's device");
        TT_FATAL(t.cache_idx->device() == device, "deepseek_v4_prefill_sdpa: cache_idx must be on q's device");
        const auto cs = cache.logical_shape();
        TT_FATAL(
            cs.rank() == 4 && cs[0] == 1 && cs[1] == 1 && cs[2] > 0 && cs[3] == K_DIM,
            "deepseek_v4_prefill_sdpa: cache_kv must be [1, 1, T, {}] with T > 0 (got {})",
            K_DIM,
            cs);
    }
    if (t.attention_sink.has_value()) {
        const auto& sink = t.attention_sink.value();
        check_row_major_dram(sink, "attention_sink");
        TT_FATAL(sink.device() == device, "deepseek_v4_prefill_sdpa: attention_sink must be on q's device");
        TT_FATAL(sink.dtype() == DataType::BFLOAT16, "deepseek_v4_prefill_sdpa: attention_sink must be bf16");
        TT_FATAL(
            sink.logical_shape() == ttnn::Shape({1, 1, 1, t.q.logical_shape()[1]}),
            "deepseek_v4_prefill_sdpa: attention_sink must be [1, 1, 1, H]");
    }
}
}  // namespace

void DeepseekV4PrefillSDPAOperation::validate_on_program_cache_hit(
    const DeepseekV4PrefillSDPAParams& /*attrs*/, const DeepseekV4PrefillSDPAInputs& t) {
    validate_non_hashed(t);
}

void DeepseekV4PrefillSDPAOperation::validate_on_program_cache_miss(
    const DeepseekV4PrefillSDPAParams& attrs, const DeepseekV4PrefillSDPAInputs& t) {
    TT_FATAL(tt::tt_metal::hal::get_arch() == tt::ARCH::BLACKHOLE, "deepseek_v4_prefill_sdpa is Blackhole-only");
    TT_FATAL(
        t.cache_kv.has_value() == t.cache_idx.has_value(),
        "deepseek_v4_prefill_sdpa: cache_kv and cache_idx must be given together");
    TT_FATAL(t.q.dtype() == DataType::BFLOAT16, "deepseek_v4_prefill_sdpa: q must be bf16");
    TT_FATAL(t.window_kv.dtype() == DataType::BFLOAT16, "deepseek_v4_prefill_sdpa: window_kv must be bf16");
    TT_FATAL(
        t.start.dtype() == DataType::UINT32 || t.start.dtype() == DataType::INT32,
        "deepseek_v4_prefill_sdpa: start must be uint32 or int32");
    validate_non_hashed(t);

    const auto qs = t.q.logical_shape();
    TT_FATAL(qs.rank() == 4 && qs[0] == 1, "deepseek_v4_prefill_sdpa: q must be [1, H, S, K_DIM] (got {})", qs);
    const uint32_t H = qs[1];
    const uint32_t S = qs[2];
    const uint32_t K_DIM = qs[3];
    constexpr uint32_t tile = tt::constants::TILE_WIDTH;
    TT_FATAL(H > 0, "deepseek_v4_prefill_sdpa: H must be > 0");
    TT_FATAL(S > 0, "deepseek_v4_prefill_sdpa: S must be > 0");
    TT_FATAL(K_DIM % tile == 0, "deepseek_v4_prefill_sdpa: K_DIM must be a multiple of {} (got {})", tile, K_DIM);
    TT_FATAL(
        attrs.v_dim > 0 && attrs.v_dim <= K_DIM && attrs.v_dim % tile == 0,
        "deepseek_v4_prefill_sdpa: v_dim must be a multiple of {} in (0, K_DIM={}] (got {})",
        tile,
        K_DIM,
        attrs.v_dim);
    TT_FATAL(attrs.window_size > 0, "deepseek_v4_prefill_sdpa: window_size must be > 0");
    TT_FATAL(
        t.window_kv.logical_shape() == ttnn::Shape({1, 1, attrs.window_size + S, K_DIM}),
        "deepseek_v4_prefill_sdpa: window_kv must be [1, 1, window_size + S = {}, {}] (got {})",
        attrs.window_size + S,
        K_DIM,
        t.window_kv.logical_shape());
    TT_FATAL(
        attrs.k_chunk_size >= tile && attrs.k_chunk_size % tile == 0,
        "deepseek_v4_prefill_sdpa: k_chunk_size must be a multiple of {} (got {})",
        tile,
        attrs.k_chunk_size);
    TT_FATAL(attrs.scale > 0.0f, "deepseek_v4_prefill_sdpa: scale must be > 0");

    const uint32_t dram_align = tt::tt_metal::hal::get_dram_alignment();
    TT_FATAL(
        (K_DIM * sizeof(uint16_t)) % dram_align == 0,
        "deepseek_v4_prefill_sdpa: q/kv row bytes must be {}B aligned",
        dram_align);
    TT_FATAL(
        (attrs.v_dim * sizeof(uint16_t)) % dram_align == 0,
        "deepseek_v4_prefill_sdpa: output row bytes must be {}B aligned",
        dram_align);
    if (t.cache_kv.has_value()) {
        TT_FATAL(t.cache_kv->dtype() == DataType::BFLOAT16, "deepseek_v4_prefill_sdpa: cache_kv must be bf16");
        const auto& idx = t.cache_idx.value();
        TT_FATAL(idx.dtype() == DataType::UINT32, "deepseek_v4_prefill_sdpa: cache_idx must be uint32");
        const auto is = idx.logical_shape();
        TT_FATAL(
            is.rank() == 4 && is[0] == 1 && is[1] == 1 && is[2] == S && is[3] > 0,
            "deepseek_v4_prefill_sdpa: cache_idx must be [1, 1, S, TOPK] (got {})",
            is);
        TT_FATAL(
            (is[3] * sizeof(uint32_t)) % dram_align == 0,
            "deepseek_v4_prefill_sdpa: cache_idx row bytes must be {}B aligned",
            dram_align);
    }
}

DeepseekV4PrefillSDPAOperation::spec_return_value_t DeepseekV4PrefillSDPAOperation::compute_output_specs(
    const DeepseekV4PrefillSDPAParams& attrs, const DeepseekV4PrefillSDPAInputs& t) {
    auto shape = t.q.logical_shape();
    shape[3] = attrs.v_dim;
    const tt::tt_metal::MemoryConfig out_mem{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    return tt::tt_metal::TensorSpec(
        shape, tt::tt_metal::TensorLayout(t.q.dtype(), tt::tt_metal::PageConfig(Layout::ROW_MAJOR), out_mem));
}

DeepseekV4PrefillSDPAOperation::tensor_return_value_t DeepseekV4PrefillSDPAOperation::create_output_tensors(
    const DeepseekV4PrefillSDPAParams& attrs, const DeepseekV4PrefillSDPAInputs& t) {
    return create_device_tensor(compute_output_specs(attrs, t), t.q.device());
}

ttsl::hash::hash_t DeepseekV4PrefillSDPAOperation::compute_program_hash(
    const DeepseekV4PrefillSDPAParams& attrs, const DeepseekV4PrefillSDPAInputs& t) {
    // cache_kv's length T is left out: it rides on the accessor's runtime args, so a longer cache reuses the
    // program. window_kv's shape follows from q's and window_size.
    return tt::tt_metal::operation::hash_operation<DeepseekV4PrefillSDPAOperation>(
        std::bit_cast<uint32_t>(attrs.scale),
        attrs.v_dim,
        attrs.window_size,
        attrs.k_chunk_size,
        attrs.compress_ratio,
        attrs.empty_tail,
        attrs.compute_kernel_config,
        t.q.logical_shape(),
        t.cache_idx.has_value() ? t.cache_idx->logical_shape() : tt::tt_metal::Shape{},
        t.attention_sink.has_value());
}

Tensor deepseek_v4_prefill_sdpa(
    const Tensor& q,
    const Tensor& window_kv,
    const Tensor& start,
    const std::optional<Tensor>& cache_kv,
    const std::optional<Tensor>& cache_idx,
    float scale,
    uint32_t v_dim,
    uint32_t window_size,
    uint32_t k_chunk_size,
    uint32_t compress_ratio,
    bool empty_tail,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    const std::optional<Tensor>& attention_sink) {
    using OperationType = ttnn::prim::DeepseekV4PrefillSDPAOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .scale = scale,
            .v_dim = v_dim,
            .window_size = window_size,
            .k_chunk_size = k_chunk_size,
            .compress_ratio = compress_ratio,
            .empty_tail = empty_tail,
            .compute_kernel_config = compute_kernel_config,
        },
        OperationType::tensor_args_t{
            .q = q,
            .window_kv = window_kv,
            .start = start,
            .cache_kv = cache_kv,
            .cache_idx = cache_idx,
            .attention_sink = attention_sink,
        });
}

}  // namespace ttnn::prim
