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
    const bool grid_packed = gs.rank() == 3;
    TT_FATAL(
        gs.rank() == 4 || grid_packed,
        "grid rank must be 4 ((N, Q, 1, P*2) or (N, Q*P, 1, 2)) or 3 ((B, Q, num_heads*stride*2)), got {}",
        gs);
    TT_FATAL(as.rank() == 3, "attn rank must be 3 (N, Q, P), got {}", as);
    const uint32_t n_total = static_cast<uint32_t>(vs[0]) * attrs.num_heads;
    TT_FATAL(
        grid_packed || static_cast<uint32_t>(gs[0]) == n_total,
        "grid's first dim (= {}) must equal value's batch times num_heads (= {})",
        static_cast<uint32_t>(gs[0]),
        n_total);

    // Reject zero-sized inputs: split_work_to_cores(grid, 0) and zero-page
    // CB creation are undefined; we'd rather fail loudly than crash deep in
    // the program factory.
    TT_FATAL(vs[0] > 0, "N must be > 0");
    TT_FATAL(vs[1] > 0 && vs[2] > 0, "h_in and w_in must be > 0");
    TT_FATAL(as[1] > 0, "Q must be > 0");
    TT_FATAL(as[2] > 0, "attn's last dim must be > 0");

    const uint32_t nh = attrs.num_heads;
    TT_FATAL(nh > 0, "num_heads must be > 0");
    TT_FATAL(
        vs[-1] % nh == 0,
        "value's last dim (= {}) must be divisible by num_heads (= {})",
        static_cast<uint32_t>(vs[-1]),
        nh);
    const uint32_t d = static_cast<uint32_t>(vs[-1]) / nh;
    // The reader scatters each D-wide value stick across ceil(D/32) tiles
    // laid side by side (16 values per face half), and the writer gathers
    // them back per query row, so any positive multiple of 16 works. A head's
    // slice starts at h*D*2 bytes, which the 16-multiple also keeps NoC-aligned.
    TT_FATAL(d > 0 && d % 16 == 0, "value's per-head D (= {}) must be a positive multiple of 16", d);

    const uint32_t q = static_cast<uint32_t>(as[1]);
    // attn is either (N, Q, P) — the head is already in the batch index — or
    // (B, Q, num_heads*stride), where the head is a byte offset into the row and this
    // call reads num_points of it starting at point_offset. The row width tells them apart:
    // the batch dim cannot, because num_heads == 1 makes N and B the same number.
    const uint32_t p = attrs.num_points > 0 ? attrs.num_points : static_cast<uint32_t>(as[2]);
    TT_FATAL(p > 0, "P must be > 0");
    const bool attn_wide = static_cast<uint32_t>(as[2]) != p;
    if (attn_wide) {
        TT_FATAL(
            static_cast<uint32_t>(as[2]) % nh == 0,
            "attn's last dim (= {}) must be divisible by num_heads (= {})",
            static_cast<uint32_t>(as[2]),
            nh);
        TT_FATAL(
            static_cast<uint32_t>(as[0]) * nh == n_total,
            "packed attn's first dim (= {}) times num_heads (= {}) must equal N (= {})",
            static_cast<uint32_t>(as[0]),
            nh,
            n_total);
    } else {
        TT_FATAL(
            static_cast<uint32_t>(as[0]) == n_total,
            "attn's first dim (= {}) must equal N (= {})",
            static_cast<uint32_t>(as[0]),
            n_total);
    }
    const uint32_t attn_head_stride = attn_wide ? static_cast<uint32_t>(as[2]) / nh : p;
    TT_FATAL(
        attrs.point_offset + p <= attn_head_stride,
        "point_offset (= {}) plus P (= {}) overruns attn's per-head run (= {})",
        attrs.point_offset,
        p,
        attn_head_stride);
    const uint32_t gw = static_cast<uint32_t>(gs[-1]);
    TT_FATAL(gw % 2 == 0 && gw > 0, "grid last dim (= {}) must be a positive multiple of 2 (x, y)", gw);
    if (grid_packed) {
        // Rank 3 packs every head and level into the row, the same way attn does.
        TT_FATAL(
            (gw / 2) % nh == 0,
            "packed grid's last dim (= {}) must hold a multiple of num_heads (= {}) points",
            gw,
            nh);
        const uint32_t grid_head_stride = (gw / 2) / nh;
        TT_FATAL(
            static_cast<uint32_t>(gs[0]) * nh == n_total,
            "packed grid's first dim (= {}) times num_heads (= {}) must equal N (= {})",
            static_cast<uint32_t>(gs[0]),
            nh,
            n_total);
        TT_FATAL(static_cast<uint32_t>(gs[1]) == q, "packed grid's Q (= {}) must equal attn Q (= {})", gs[1], q);
        TT_FATAL(
            attrs.point_offset + p <= grid_head_stride,
            "point_offset (= {}) plus P (= {}) overruns the grid's per-head run (= {})",
            attrs.point_offset,
            p,
            grid_head_stride);
    } else {
        // A ROW_MAJOR page is the last dimension, so the point axis is left to the caller: folded
        // into the page it is one NoC read per query, spelled out it is P reads of four bytes and a
        // rewrite on the caller's side to produce them. Any divisor of P in between is legal.
        TT_FATAL(p % (gw / 2) == 0, "grid last dim (= {}) must hold a divisor of P (= {}) points", gw, p);
        TT_FATAL(
            static_cast<uint32_t>(gs[1]) * (gw / 2) == q * p,
            "grid holds {} points, attn expects Q*P = {}",
            static_cast<uint32_t>(gs[1]) * (gw / 2),
            q * p);
        TT_FATAL(gs[-2] == 1, "grid 3rd dim must be 1 (single sample row per query)");
    }
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
    const uint32_t N = vs[0] * attrs.num_heads;
    const uint32_t D = vs[3] / attrs.num_heads;
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
    bool align_corners,
    uint32_t num_heads,
    uint32_t num_points,
    uint32_t point_offset) {
    using OperationType = ttnn::operations::experimental::multi_scale_deformable_attn::MSDAOperation;
    auto attrs = OperationType::operation_attributes_t{
        .output_memory_config = memory_config.value_or(value.memory_config()),
        .align_corners = align_corners,
        .num_heads = num_heads,
        .num_points = num_points,
        .point_offset = point_offset,
    };
    auto args = OperationType::tensor_args_t{
        .value = value,
        .grid = grid,
        .attn = attn,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, args);
}

}  // namespace ttnn::prim
