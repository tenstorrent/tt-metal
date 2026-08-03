// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_scale_deformable_attn_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::multi_scale_deformable_attn {

void MSDAOperation::validate_on_program_cache_miss(const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto& value = args.value;
    const auto& grid = args.grid;
    const auto& attn = args.attn;

    TT_FATAL(value.storage_type() == StorageType::DEVICE, "value must be on device");
    TT_FATAL(grid.storage_type() == StorageType::DEVICE, "grid must be on device");
    TT_FATAL(attn.storage_type() == StorageType::DEVICE, "attn must be on device");
    TT_FATAL(
        value.device() == grid.device() && value.device() == attn.device(),
        "value, grid, and attn must be on the same device");

    TT_FATAL(value.dtype() == DataType::BFLOAT16, "value must be BFLOAT16");
    TT_FATAL(grid.dtype() == DataType::BFLOAT16, "grid must be BFLOAT16");
    TT_FATAL(attn.dtype() == DataType::BFLOAT16, "attn must be BFLOAT16");

    TT_FATAL(value.layout() == Layout::ROW_MAJOR, "value must be ROW_MAJOR");
    TT_FATAL(grid.layout() == Layout::ROW_MAJOR, "grid must be ROW_MAJOR");
    TT_FATAL(attn.layout() == Layout::ROW_MAJOR, "attn must be ROW_MAJOR");

    // Reader/writer use per-stick indexing via TensorAccessor which assumes
    // interleaved DRAM/L1; sharded layouts would mis-address. TODO: support
    // sharded inputs (would require reader/writer to use shard-aware
    // accessor or per-core offset tables).
    using tt::tt_metal::TensorMemoryLayout;
    TT_FATAL(
        value.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "value memory_layout must be INTERLEAVED");
    TT_FATAL(
        grid.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "grid memory_layout must be INTERLEAVED");
    TT_FATAL(
        attn.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "attn memory_layout must be INTERLEAVED");
    TT_FATAL(
        attrs.output_memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "output_memory_config memory_layout must be INTERLEAVED");

    const auto& vs = value.logical_shape();
    const auto& gs = grid.logical_shape();
    const auto& as = attn.logical_shape();

    TT_FATAL(vs.rank() == 4, "value rank must be 4 (N, h_in, w_in, D), got {}", vs);
    TT_FATAL(gs.rank() == 4, "grid rank must be 4 (N, Q*P, 1, 2), got {}", gs);
    TT_FATAL(as.rank() == 3, "attn rank must be 3 (N, Q, P), got {}", as);
    TT_FATAL(gs[-1] == 2, "grid last dim must be 2 (x, y)");
    TT_FATAL(gs[-2] == 1, "grid 3rd dim must be 1 (single sample per row)");
    TT_FATAL(vs[0] == gs[0] && vs[0] == as[0], "N (batch*head) dim mismatch");

    // Reject zero-sized inputs: split_work_to_cores(grid, 0) and zero-page
    // CB creation are undefined; we'd rather fail loudly than crash deep in
    // the program factory.
    TT_FATAL(vs[0] > 0, "N must be > 0");
    TT_FATAL(vs[1] > 0 && vs[2] > 0, "h_in and w_in must be > 0");
    TT_FATAL(as[1] > 0, "Q must be > 0");
    TT_FATAL(as[2] > 0, "P must be > 0");

    // TODO: generalize the kernel to support arbitrary D (multiple of 16). The
    // reader/writer currently assume D=32: a single tile row is split into
    // exactly two 32-byte halves placed in TL+TR (or BL+BR) faces, and
    // HALF_STICK_NBYTES is hardcoded to 32 in the kernels. Supporting D=64
    // or D=16 means deriving the per-row layout from element_size + D and
    // looping over multiple (half-)faces per row.
    TT_FATAL(vs[-1] == 32, "value's last dim (D) must be 32, got {}", static_cast<uint32_t>(vs[-1]));

    const uint32_t qp = static_cast<uint32_t>(gs[1]);
    const uint32_t q = static_cast<uint32_t>(as[1]);
    const uint32_t p = static_cast<uint32_t>(as[2]);
    TT_FATAL(qp == q * p, "grid dim 1 (= Q*P = {}) must equal attn Q*P (= {} * {} = {})", qp, q, p, q * p);
}

void MSDAOperation::validate_on_program_cache_hit(const operation_attributes_t& attrs, const tensor_args_t& args) {
    validate_on_program_cache_miss(attrs, args);
}

// TODO: support an `output_layout` kwarg. The current writer emits a
// ROW_MAJOR stick per query, matching the convention used by the sibling
// ttnn.grid_sample op. TILE output would require tilizing in the writer
// kernel (or a follow-up to_layout op on the caller side), which is a
// non-trivial rewrite — left for a future PR.
MSDAOperation::spec_return_value_t MSDAOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto& vs = args.value.logical_shape();  // (N, h, w, D)
    const auto& as = args.attn.logical_shape();   // (N, Q, P)
    const uint32_t N = vs[0];
    const uint32_t D = vs[3];
    const uint32_t Q = as[1];

    Shape out_shape({N, Q, D});
    return tt::tt_metal::TensorSpec(
        out_shape,
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), attrs.output_memory_config));
}

MSDAOperation::tensor_return_value_t MSDAOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    return create_device_tensor(compute_output_specs(attrs, args), args.value.device());
}

}  // namespace ttnn::operations::experimental::multi_scale_deformable_attn

namespace ttnn::prim {

ttnn::Tensor multi_scale_deformable_attn(
    const Tensor& value,
    const Tensor& grid,
    const Tensor& attn,
    const std::optional<MemoryConfig>& memory_config,
    bool align_corners) {
    using OperationType = ttnn::operations::experimental::multi_scale_deformable_attn::MSDAOperation;
    auto attrs = OperationType::operation_attributes_t{
        .output_memory_config = memory_config.value_or(value.memory_config()),
        .align_corners = align_corners,
    };
    auto args = OperationType::tensor_args_t{
        .value = value,
        .grid = grid,
        .attn = attn,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, args);
}

}  // namespace ttnn::prim
