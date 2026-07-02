// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kv_sdpa_device_operation.hpp"

#include "tt-metalium/constants.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::kv_sdpa {

using namespace tt::tt_metal;

KvSdpaDeviceOperation::program_factory_t KvSdpaDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return FlashFused{};
}

void KvSdpaDeviceOperation::validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t& ta) {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;
    TT_FATAL(
        ta.q.layout() == Layout::TILE && ta.k.layout() == Layout::TILE && ta.v.layout() == Layout::TILE,
        "kv_sdpa: q/k/v must be TILE layout");
    const auto& qs = ta.q.padded_shape();
    const auto& ks = ta.k.padded_shape();
    const auto& vs = ta.v.padded_shape();
    TT_FATAL(qs.rank() == 4 && ks.rank() == 4 && vs.rank() == 4, "kv_sdpa: q/k/v must be rank-4 [1, NH, S, DH]");
    TT_FATAL(qs[0] == 1, "kv_sdpa: batch must be 1 (got {})", qs[0]);
    const uint32_t NQH = qs[1], NKH = ks[1];
    TT_FATAL(NKH >= 1 && NQH % NKH == 0, "kv_sdpa: NQH ({}) must be a multiple of NKH ({})", NQH, NKH);
    TT_FATAL(qs[2] == TILE_HEIGHT, "kv_sdpa: query length must be exactly one tile ({}); got {}", TILE_HEIGHT, qs[2]);
    TT_FATAL(qs[3] == ks[3] && qs[3] == vs[3], "kv_sdpa: head_dim must match across q/k/v");
    TT_FATAL(qs[3] % TILE_WIDTH == 0, "kv_sdpa: head_dim ({}) must be tile-aligned", qs[3]);
    TT_FATAL(ks[2] % TILE_HEIGHT == 0 && ks[2] == vs[2], "kv_sdpa: kv length must be tile-aligned and match k/v");
    if (ta.past_k.has_value()) {
        TT_FATAL(ta.past_v.has_value(), "kv_sdpa: past_k and past_v must be provided together");
        const auto& pks = ta.past_k->padded_shape();
        const auto& pvs = ta.past_v->padded_shape();
        TT_FATAL(
            ta.past_k->layout() == Layout::TILE && ta.past_v->layout() == Layout::TILE,
            "kv_sdpa: past_k/past_v must be TILE layout");
        TT_FATAL(pks.rank() == 4 && pvs.rank() == 4, "kv_sdpa: past_k/past_v must be rank-4");
        TT_FATAL(pks[1] == NKH && pvs[1] == NKH, "kv_sdpa: past_k/past_v NKH must match k/v");
        TT_FATAL(pks[3] == qs[3] && pvs[3] == qs[3], "kv_sdpa: past_k/past_v head_dim must match");
        TT_FATAL(
            pks[2] % TILE_HEIGHT == 0 && pks[2] == pvs[2], "kv_sdpa: prefix length must be tile-aligned and match");
        TT_FATAL(ta.past_k->dtype() == ta.k.dtype(), "kv_sdpa: past_k/k dtype must match (shared reader CB)");
    }
    if (ta.mask.has_value()) {
        const auto& ms = ta.mask->padded_shape();
        const uint32_t prefix = ta.past_k.has_value() ? ta.past_k->padded_shape()[2] : 0;
        const uint32_t kv_total = prefix + ks[2];
        TT_FATAL(ta.mask->layout() == Layout::TILE, "kv_sdpa: attn_mask must be TILE layout");
        TT_FATAL(ms.rank() == 4, "kv_sdpa: attn_mask must be rank-4 [1, 1, Sq, KV]");
        // Mask is broadcast across Q heads (dim 1) and shares the single Sq tile-row; its KV (last)
        // dim must cover the full folded [prefix ; suffix] KV so column-tile g aligns with KV-tile g.
        TT_FATAL(ms[2] == TILE_HEIGHT, "kv_sdpa: attn_mask Sq must be one tile ({}); got {}", TILE_HEIGHT, ms[2]);
        TT_FATAL(ms[3] == kv_total, "kv_sdpa: attn_mask KV ({}) must equal prefix+suffix KV ({})", ms[3], kv_total);
        TT_FATAL(ms[3] % TILE_WIDTH == 0, "kv_sdpa: attn_mask KV ({}) must be tile-aligned", ms[3]);
    }
}

KvSdpaDeviceOperation::spec_return_value_t KvSdpaDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& ta) {
    // Output matches q's shape/dtype, interleaved DRAM (consumed by the downstream o-projection, whose
    // concat-heads matmul_decode reader reshards an interleaved input).
    return TensorSpec(
        ta.q.logical_shape(),
        TensorLayout(
            ta.q.dtype(),
            PageConfig(Layout::TILE, ta.q.tensor_spec().tile()),
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt)));
}

KvSdpaDeviceOperation::tensor_return_value_t KvSdpaDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& ta) {
    return create_device_tensor(compute_output_specs(attrs, ta), ta.q.device());
}

}  // namespace ttnn::operations::kv_sdpa

namespace ttnn::prim {
ttnn::operations::kv_sdpa::KvSdpaDeviceOperation::tensor_return_value_t kv_sdpa(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    std::optional<Tensor> mask,
    uint32_t scale_bits,
    std::optional<Tensor> past_k,
    std::optional<Tensor> past_v,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    using Op = ttnn::operations::kv_sdpa::KvSdpaDeviceOperation;
    auto attrs = Op::operation_attributes_t{.scale_bits = scale_bits, .compute_kernel_config = compute_kernel_config};
    auto args = Op::tensor_args_t{
        .q = q, .k = k, .v = v, .mask = std::move(mask), .past_k = std::move(past_k), .past_v = std::move(past_v)};
    return ttnn::device_operation::launch<Op>(attrs, args);
}
}  // namespace ttnn::prim
