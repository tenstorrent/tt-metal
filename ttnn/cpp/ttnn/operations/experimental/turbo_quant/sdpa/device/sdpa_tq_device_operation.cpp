// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_tq_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::turbo_quant {

void SDPATQDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    // Q: BF16, TILE, on device
    TT_FATAL(args.q.storage_type() == StorageType::DEVICE, "Q must be on device");
    TT_FATAL(args.q.layout() == Layout::TILE, "Q must be TILE layout");
    TT_FATAL(args.q.dtype() == tt::tt_metal::DataType::BFLOAT16, "Q must be BF16");

    // K/V indices: BFP4, BFP8, or BF16
    TT_FATAL(
        args.k_indices.dtype() == tt::tt_metal::DataType::BFLOAT4_B ||
            args.k_indices.dtype() == tt::tt_metal::DataType::BFLOAT8_B ||
            args.k_indices.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "K indices must be BFP4_B, BFP8_B, or BF16");
    TT_FATAL(
        args.v_indices.dtype() == tt::tt_metal::DataType::BFLOAT4_B ||
            args.v_indices.dtype() == tt::tt_metal::DataType::BFLOAT8_B ||
            args.v_indices.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "V indices must be BFP4_B, BFP8_B, or BF16");
    TT_FATAL(args.k_indices.layout() == Layout::TILE, "K indices must be TILE layout");
    TT_FATAL(args.v_indices.layout() == Layout::TILE, "V indices must be TILE layout");

    // Norms: BF16 or BFP8_B, TILE. BFP8 halves on-device norms storage; the
    // compute kernel typecasts BFP8 → BF16 before the bcast_cols multiply.
    TT_FATAL(
        args.k_norms.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            args.k_norms.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "K norms must be BF16 or BFP8_B");
    TT_FATAL(
        args.v_norms.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            args.v_norms.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "V norms must be BF16 or BFP8_B");

    // Centroids
    TT_FATAL(attrs.centroids.size() >= 2 && attrs.centroids.size() <= 16, "Need 2-16 centroids");

    // Sliding-window fused hybrid (Phase 1): when recent_window > 0, the ring
    // tensors must be provided and have compatible dtypes. Phase 1 only plumbs
    // them through; the reader / compute kernel still ignore them. Phase 2/3
    // wires the per-chunk source branch.
    if (attrs.recent_window > 0) {
        TT_FATAL(
            args.k_ring.has_value() && args.v_ring.has_value() && args.ring_page_table.has_value(),
            "recent_window > 0 requires k_ring, v_ring, ring_page_table to be set");
        TT_FATAL(
            args.k_ring->dtype() == tt::tt_metal::DataType::BFLOAT4_B ||
                args.k_ring->dtype() == tt::tt_metal::DataType::BFLOAT8_B ||
                args.k_ring->dtype() == tt::tt_metal::DataType::BFLOAT16,
            "k_ring must be BFP4_B, BFP8_B, or BF16");
        TT_FATAL(
            args.v_ring->dtype() == tt::tt_metal::DataType::BFLOAT4_B ||
                args.v_ring->dtype() == tt::tt_metal::DataType::BFLOAT8_B ||
                args.v_ring->dtype() == tt::tt_metal::DataType::BFLOAT16,
            "v_ring must be BFP4_B, BFP8_B, or BF16");
        TT_FATAL(args.k_ring->layout() == Layout::TILE, "k_ring must be TILE layout");
        TT_FATAL(args.v_ring->layout() == Layout::TILE, "v_ring must be TILE layout");
        TT_FATAL(
            args.ring_page_table->dtype() == tt::tt_metal::DataType::INT32 ||
                args.ring_page_table->dtype() == tt::tt_metal::DataType::UINT32,
            "ring_page_table must be Int32 or UInt32");
        // Mutually exclusive with cross-core merge for Phase 1 to keep the
        // initial implementation focused. Lifting this requires cb_lse_out
        // de-aliasing similar to the return_lse case (cb_merge_new_max == c_3).
        TT_FATAL(attrs.num_cores_per_head == 1, "recent_window > 0 currently incompatible with num_cores_per_head > 1");
        TT_FATAL(
            !attrs.pre_rescaled, "recent_window > 0 currently incompatible with pre_rescaled (kernel chooses both)");
        // Hybrid mode repurposes the unused Tier-2A CBs (c_18, c_19) as the
        // reader's ring K / V data CBs. Forcing num_cores_per_head == 1 above
        // guarantees the Tier-2A reducer / worker pack-and-skip paths are
        // dormant and these CBs are free.
    }
}

SDPATQDeviceOperation::spec_return_value_t SDPATQDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    spec_return_value_t specs;
    // [0] Output: same shape as Q, BF16
    specs.emplace_back(
        args.q.logical_shape(),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::PageConfig(args.q.layout()), attrs.output_mem_config));
    if (attrs.return_lse) {
        // [1] LSE: shape [B, NQH, 1, TILE_W=32], 1 tile per (B, NQH) entry.
        // Kernel packs one BF16 LSE value per tile (col 0 of row 0); rest is
        // padding from the broadcast pack. Host-side combine reads col 0.
        const auto& q_shape = args.q.logical_shape();
        constexpr uint32_t TILE_W = 32;
        ttnn::Shape lse_shape({q_shape[0], q_shape[1], q_shape[2], TILE_W});
        specs.emplace_back(
            lse_shape,
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::PageConfig(args.q.layout()), attrs.output_mem_config));
    }
    return specs;
}

SDPATQDeviceOperation::tensor_return_value_t SDPATQDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    auto specs = compute_output_specs(attrs, args);
    tensor_return_value_t outs;
    outs.reserve(specs.size());
    for (const auto& spec : specs) {
        outs.push_back(create_device_tensor(spec, args.q.device()));
    }
    return outs;
}

}  // namespace ttnn::operations::experimental::turbo_quant

namespace ttnn::prim {

std::vector<Tensor> turbo_quant_sdpa_decode(
    const Tensor& q,
    const Tensor& k_indices,
    const Tensor& k_norms,
    const Tensor& v_indices,
    const Tensor& v_norms,
    const Tensor& page_table,
    const Tensor& cur_pos,
    const std::vector<float>& centroids,
    float scale,
    bool pre_rescaled,
    uint32_t num_cores_per_head,
    bool return_lse,
    uint32_t recent_window,
    const std::optional<Tensor>& k_ring,
    const std::optional<Tensor>& v_ring,
    const std::optional<Tensor>& ring_page_table) {
    using Op = ttnn::operations::experimental::turbo_quant::SDPATQDeviceOperation;
    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            scale, centroids, pre_rescaled, ttnn::MemoryConfig{}, num_cores_per_head, return_lse, recent_window},
        Op::tensor_args_t{
            q, k_indices, k_norms, v_indices, v_norms, page_table, cur_pos, k_ring, v_ring, ring_page_table});
}

}  // namespace ttnn::prim
