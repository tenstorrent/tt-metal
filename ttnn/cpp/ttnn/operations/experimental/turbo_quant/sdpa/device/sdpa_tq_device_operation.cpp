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

    // K/V indices: BFP4, TILE
    TT_FATAL(args.k_indices.dtype() == tt::tt_metal::DataType::BFLOAT4_B, "K indices must be BFP4_B");
    TT_FATAL(args.v_indices.dtype() == tt::tt_metal::DataType::BFLOAT4_B, "V indices must be BFP4_B");
    TT_FATAL(args.k_indices.layout() == Layout::TILE, "K indices must be TILE layout");
    TT_FATAL(args.v_indices.layout() == Layout::TILE, "V indices must be TILE layout");

    // Norms: BF16, TILE
    TT_FATAL(args.k_norms.dtype() == tt::tt_metal::DataType::BFLOAT16, "K norms must be BF16");
    TT_FATAL(args.v_norms.dtype() == tt::tt_metal::DataType::BFLOAT16, "V norms must be BF16");

    // Centroids
    TT_FATAL(attrs.centroids.size() >= 2 && attrs.centroids.size() <= 16, "Need 2-16 centroids");
}

SDPATQDeviceOperation::spec_return_value_t SDPATQDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    // Output: same shape as Q, BF16
    return TensorSpec(
        args.q.logical_shape(),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::PageConfig(args.q.layout()), attrs.output_mem_config));
}

SDPATQDeviceOperation::tensor_return_value_t SDPATQDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    auto output_spec = compute_output_specs(attrs, args);
    return create_device_tensor(output_spec, args.q.device());
}

}  // namespace ttnn::operations::experimental::turbo_quant

namespace ttnn::prim {

Tensor turbo_quant_sdpa_decode(
    const Tensor& q,
    const Tensor& k_indices,
    const Tensor& k_norms,
    const Tensor& v_indices,
    const Tensor& v_norms,
    const Tensor& page_table,
    const Tensor& cur_pos,
    const std::vector<float>& centroids,
    float scale) {
    using Op = ttnn::operations::experimental::turbo_quant::SDPATQDeviceOperation;
    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{scale, centroids, ttnn::MemoryConfig{}},
        Op::tensor_args_t{q, k_indices, k_norms, v_indices, v_norms, page_table, cur_pos});
}

}  // namespace ttnn::prim
