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

    // Tiny-tile support: the query tile-row may be shorter than 32 (tiny tile), but the tile width
    // must stay 32 and blocked (face-packed) dtypes cannot use tiny heights. All tile math below is
    // derived from the tensors' actual tiles rather than the global 32x32 constants.
    const auto q_tile = ta.q.tensor_spec().tile();
    const auto k_tile = ta.k.tensor_spec().tile();
    const auto v_tile = ta.v.tensor_spec().tile();
    const uint32_t q_tile_h = q_tile.get_height();
    const uint32_t k_tile_h = k_tile.get_height();
    const uint32_t v_tile_h = v_tile.get_height();
    for (const auto& [name, tile] : {std::pair{"q", q_tile}, std::pair{"k", k_tile}, std::pair{"v", v_tile}}) {
        TT_FATAL(
            tile.get_width() == TILE_WIDTH,
            "kv_sdpa: {} tile width must be {}, got {}",
            name,
            TILE_WIDTH,
            tile.get_width());
    }

    TT_FATAL(qs[2] == q_tile_h, "kv_sdpa: query length must be exactly one tile ({}); got {}", q_tile_h, qs[2]);
    TT_FATAL(qs[3] == ks[3] && qs[3] == vs[3], "kv_sdpa: head_dim must match across q/k/v");
    TT_FATAL(qs[3] % TILE_WIDTH == 0, "kv_sdpa: head_dim ({}) must be tile-aligned", qs[3]);
    TT_FATAL(
        ks[2] % k_tile_h == 0 && vs[2] % v_tile_h == 0 && ks[2] == vs[2],
        "kv_sdpa: kv length must be tile-aligned and match k/v");
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
        const uint32_t pk_tile_h = ta.past_k->tensor_spec().tile().get_height();
        const uint32_t pv_tile_h = ta.past_v->tensor_spec().tile().get_height();
        TT_FATAL(
            pks[2] % pk_tile_h == 0 && pvs[2] % pv_tile_h == 0 && pks[2] == pvs[2],
            "kv_sdpa: prefix length must be tile-aligned and match");
        // Prefix and suffix do NOT share a reader CB: the factory derives the prefix format
        // independently (pkdf) and allocates its own CB pair (c_8/c_9) with its own tile size, the
        // reader takes per-source tile bytes (get_tile_size(cb_k_prefix) vs cb_k_in) and issues a
        // separate READ_KV_SOURCE per phase, and the compute reconfigures the unpack format per phase
        // from whichever K CB it is handed. So the dtypes may legitimately differ -- e.g. a bfloat4_b
        // resident prefix (halving the dominant prefix-KV read traffic) with bfloat8_b suffix
        // activations. Only require that past_v matches past_k, since they share the phase geometry.
        TT_FATAL(
            ta.past_v->dtype() == ta.past_k->dtype(),
            "kv_sdpa: past_k/past_v dtype must match (same prefix phase geometry)");
    }
    if (ta.mask.has_value()) {
        const auto& ms = ta.mask->padded_shape();
        const uint32_t prefix = ta.past_k.has_value() ? ta.past_k->padded_shape()[2] : 0;
        const uint32_t kv_total = prefix + ks[2];
        TT_FATAL(ta.mask->layout() == Layout::TILE, "kv_sdpa: attn_mask must be TILE layout");
        TT_FATAL(ms.rank() == 4, "kv_sdpa: attn_mask must be rank-4 [1, 1, Sq, KV]");
        // The mask is broadcast across Q heads (dim 1) and shares the single Sq tile-row. It is added to
        // the QK score tiles, so its tile HEIGHT must match Q's (the score tile is [Sq_h x 32]) -- at a
        // tiny tile that is 16, not 32.
        const uint32_t mask_tile_h = ta.mask->tensor_spec().tile().get_height();
        const uint32_t q_tile_h = ta.q.tensor_spec().tile().get_height();
        TT_FATAL(
            mask_tile_h == q_tile_h,
            "kv_sdpa: attn_mask tile height ({}) must match q's ({}) -- the mask is added to the "
            "[Sq_h x 32] score tiles",
            mask_tile_h,
            q_tile_h);
        TT_FATAL(ms[2] == mask_tile_h, "kv_sdpa: attn_mask Sq must be one tile ({}); got {}", mask_tile_h, ms[2]);
        // Column-tile g of the mask must align with KV-tile g of the folded [prefix ; suffix] KV, which
        // the reader relies on to index the mask with the same tile counter it uses for K/V. Mask column
        // tiles are TILE_WIDTH wide, so the prefix must occupy a whole number of them AND the prefix's
        // own tile height must be TILE_WIDTH -- otherwise prefix tile g and mask column-tile g cover
        // different KV ranges. (A 32x32 prefix + a 16x32 suffix, the pi0.5 config, satisfies this: the
        // suffix is a single K tile at both tile heights.)
        TT_FATAL(prefix % TILE_WIDTH == 0, "kv_sdpa: attn_mask requires a tile-aligned prefix; got {}", prefix);
        if (ta.past_k.has_value()) {
            const uint32_t pk_tile_h = ta.past_k->tensor_spec().tile().get_height();
            TT_FATAL(
                pk_tile_h == TILE_WIDTH,
                "kv_sdpa: attn_mask requires a {}-row prefix tile so prefix tile g aligns with mask "
                "column-tile g; got {}",
                TILE_WIDTH,
                pk_tile_h);
        }
        // The mask may be padded up to a tile boundary beyond the real KV (e.g. a 16-row suffix inside a
        // 32-wide column tile), so require coverage rather than exact equality.
        TT_FATAL(ms[3] % TILE_WIDTH == 0, "kv_sdpa: attn_mask KV ({}) must be tile-aligned", ms[3]);
        TT_FATAL(ms[3] >= kv_total, "kv_sdpa: attn_mask KV ({}) must cover prefix+suffix KV ({})", ms[3], kv_total);
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
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    uint32_t max_kv_chunk_tiles,
    uint32_t kv_splits,
    std::vector<uint32_t> prefix_valid_tiles) {
    using Op = ttnn::operations::kv_sdpa::KvSdpaDeviceOperation;
    auto attrs = Op::operation_attributes_t{
        .scale_bits = scale_bits,
        .compute_kernel_config = compute_kernel_config,
        .max_kv_chunk_tiles = max_kv_chunk_tiles,
        .kv_splits = kv_splits,
        .prefix_valid_tiles = std::move(prefix_valid_tiles)};
    auto args = Op::tensor_args_t{
        .q = q, .k = k, .v = v, .mask = std::move(mask), .past_k = std::move(past_k), .past_v = std::move(past_v)};
    return ttnn::device_operation::launch<Op>(attrs, args);
}
}  // namespace ttnn::prim
