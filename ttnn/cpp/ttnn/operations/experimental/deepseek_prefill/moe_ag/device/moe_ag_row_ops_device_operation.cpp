// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_ag_row_ops_device_operation.hpp"

#include "moe_ag_common.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;
using namespace ttnn::operations::experimental::deepseek_prefill::moe_ag;

namespace ttnn::prim {

namespace {

TensorSpec bf16_spec(ttnn::Shape shape, Layout layout) {
    return TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(layout), DRAM_MEMORY_CONFIG));
}

void check_output(const std::optional<Tensor>& out, const TensorSpec& spec, const char* op) {
    if (!out.has_value()) {
        return;
    }
    check_dram_interleaved(*out, op, "preallocated_output");
    TT_FATAL(
        out->logical_shape() == spec.logical_shape() && out->layout() == spec.layout() &&
            out->dtype() == spec.data_type(),
        "{}: preallocated_output must be {} {} {}, got {} {} {}",
        op,
        spec.data_type(),
        spec.layout(),
        spec.logical_shape(),
        out->dtype(),
        out->layout(),
        out->logical_shape());
}

Tensor output_or_new(const std::optional<Tensor>& out, const TensorSpec& spec, const Tensor& like) {
    return out.has_value() ? *out : create_device_tensor(spec, like.device());
}

}  // namespace

// ---------------------------------------------------------------- sum_rows_tiled
void MoeAgSumRowsTiledDeviceOperation::validate_on_program_cache_miss(
    const MoeAgSumRowsTiledParams& args, const MoeAgSumRowsTiledInputs& t) {
    constexpr const char* op = "moe_ag_sum_rows_tiled";
    check_row_major(t.src, DataType::BFLOAT16, op, "src");
    const uint32_t H = t.src.logical_shape()[-1];
    TT_FATAL(H % 1024 == 0, "{}: hidden {} must be a multiple of 1024", op, H);
    TT_FATAL(
        args.num_rows >= 32 && args.num_rows % 32 == 0, "{}: num_rows {} must be a multiple of 32", op, args.num_rows);
    TT_FATAL(args.num_blocks >= 2, "{}: num_blocks {} must be >= 2", op, args.num_blocks);
    TT_FATAL(
        (args.num_blocks - 1) * args.block_stride + args.num_rows <= rm_rows(t.src),
        "{}: {} blocks of {} rows at stride {} exceed src rows {}",
        op,
        args.num_blocks,
        args.num_rows,
        args.block_stride,
        rm_rows(t.src));
    check_output(t.preallocated_output, compute_output_specs(args, t), op);
}

TensorSpec MoeAgSumRowsTiledDeviceOperation::compute_output_specs(
    const MoeAgSumRowsTiledParams& args, const MoeAgSumRowsTiledInputs& t) {
    return bf16_spec(ttnn::Shape({1, 1, args.num_rows, t.src.logical_shape()[-1]}), Layout::TILE);
}

Tensor MoeAgSumRowsTiledDeviceOperation::create_output_tensors(
    const MoeAgSumRowsTiledParams& args, const MoeAgSumRowsTiledInputs& t) {
    return output_or_new(t.preallocated_output, compute_output_specs(args, t), t.src);
}

Tensor moe_ag_sum_rows_tiled(
    const Tensor& src,
    uint32_t num_rows,
    uint32_t num_blocks,
    uint32_t block_stride,
    const std::optional<Tensor>& preallocated_output) {
    using Op = MoeAgSumRowsTiledDeviceOperation;
    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{.num_rows = num_rows, .num_blocks = num_blocks, .block_stride = block_stride},
        Op::tensor_args_t{.src = src, .preallocated_output = preallocated_output});
}

// ---------------------------------------------------------------- add_rows
void MoeAgAddRowsDeviceOperation::validate_on_program_cache_miss(
    const MoeAgAddRowsParams& args, const MoeAgAddRowsInputs& t) {
    constexpr const char* op = "moe_ag_add_rows";
    check_row_major(t.a, DataType::BFLOAT16, op, "a");
    check_row_major(t.b, DataType::BFLOAT16, op, "b");
    check_row_major(t.chip_info, DataType::UINT32, op, "chip_info");
    TT_FATAL(t.chip_info.logical_shape()[-1] == 16, "{}: chip_info must be [.., 1, 16] uint32", op);
    const uint32_t H = t.a.logical_shape()[-1];
    TT_FATAL(H % 1024 == 0 && t.b.logical_shape()[-1] == H, "{}: a / b width must match, a multiple of 1024", op);
    TT_FATAL(args.num_rows >= 1, "{}: num_rows must be >= 1", op);
    TT_FATAL(args.batch >= 1, "{}: batch must be >= 1", op);
    TT_FATAL(args.a_offset + args.num_rows <= rm_rows(t.a), "{}: a rows out of range", op);
    TT_FATAL(args.info_offset || args.b_offset + args.num_rows <= rm_rows(t.b), "{}: b rows out of range", op);
    check_output(t.preallocated_output, compute_output_specs(args, t), op);
}

TensorSpec MoeAgAddRowsDeviceOperation::compute_output_specs(
    const MoeAgAddRowsParams& args, const MoeAgAddRowsInputs& t) {
    return bf16_spec(ttnn::Shape({1, 1, args.num_rows, t.a.logical_shape()[-1]}), Layout::ROW_MAJOR);
}

Tensor MoeAgAddRowsDeviceOperation::create_output_tensors(const MoeAgAddRowsParams& args, const MoeAgAddRowsInputs& t) {
    return output_or_new(t.preallocated_output, compute_output_specs(args, t), t.a);
}

Tensor moe_ag_add_rows(
    const Tensor& a,
    const Tensor& b,
    const Tensor& chip_info,
    uint32_t num_rows,
    uint32_t a_offset,
    uint32_t b_offset,
    bool info_offset,
    const std::optional<Tensor>& preallocated_output) {
    using Op = MoeAgAddRowsDeviceOperation;
    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            .num_rows = num_rows, .a_offset = a_offset, .b_offset = b_offset, .info_offset = info_offset},
        Op::tensor_args_t{.a = a, .b = b, .chip_info = chip_info, .preallocated_output = preallocated_output});
}

// ---------------------------------------------------------------- untilize_active
void MoeAgUntilizeActiveDeviceOperation::validate_on_program_cache_miss(
    const MoeAgUntilizeActiveParams& args, const MoeAgUntilizeActiveInputs& t) {
    constexpr const char* op = "moe_ag_untilize_active";
    check_dram_interleaved(t.y, op, "y");
    TT_FATAL(t.y.layout() == Layout::TILE && t.y.dtype() == DataType::BFLOAT8_B, "{}: y must be bfloat8_b TILE", op);
    for (const auto* p : {&t.counts, &t.regions, &t.local_slot_map}) {
        check_row_major(*p, DataType::UINT32, op, "counts / regions / local_slot_map");
        check_single_row(*p, op, "counts / regions / local_slot_map");
    }
    const uint32_t NG = t.local_slot_map.logical_shape()[-1];
    TT_FATAL(
        t.counts.logical_shape()[-1] == NG && t.regions.logical_shape()[-1] == NG,
        "{}: counts / regions / local_slot_map must all be [1, NG]",
        op);
    const uint32_t H = t.y.logical_shape()[-1], W = args.tiles_per_block;
    TT_FATAL(W >= 1 && H % (32 * W) == 0, "{}: hidden {} must be a multiple of 32 x tiles_per_block {}", op, H, W);
    // the writer stores whole 32-row tiles: a ragged row count would let the last tile run past the output
    // (per matrix: the tiles of each leading-dim slice are padded separately, the reader walks them as one matrix)
    TT_FATAL(t.y.logical_shape()[-2] % 32 == 0, "{}: y rows {} must be a multiple of 32", op, t.y.logical_shape()[-2]);
    // counts / regions / the slot map are read to L1 at NG x 4 B strides: Blackhole DRAM reads need the L1 and DRAM
    // offsets equal modulo 64 B, so NG x 4 must be a multiple of 64
    TT_FATAL(NG % 16 == 0, "{}: the number of global experts {} must be a multiple of 16", op, NG);
    TT_FATAL(
        args.experts_per_chip >= 1 && args.experts_per_chip <= NG,
        "{}: experts_per_chip {} must be in [1, {}]",
        op,
        args.experts_per_chip,
        NG);
    check_output(t.preallocated_output, compute_output_specs(args, t), op);
}

TensorSpec MoeAgUntilizeActiveDeviceOperation::compute_output_specs(
    const MoeAgUntilizeActiveParams&, const MoeAgUntilizeActiveInputs& t) {
    const auto& s = t.y.logical_shape();
    uint32_t rows = 1;
    for (int i = 0; i + 1 < static_cast<int>(s.rank()); ++i) {
        rows *= s[i];
    }
    return bf16_spec(ttnn::Shape({rows, s[-1]}), Layout::ROW_MAJOR);
}

Tensor MoeAgUntilizeActiveDeviceOperation::create_output_tensors(
    const MoeAgUntilizeActiveParams& args, const MoeAgUntilizeActiveInputs& t) {
    return output_or_new(t.preallocated_output, compute_output_specs(args, t), t.y);
}

Tensor moe_ag_untilize_active(
    const Tensor& y,
    const Tensor& counts,
    const Tensor& regions,
    const Tensor& local_slot_map,
    uint32_t experts_per_chip,
    uint32_t tiles_per_block,
    const std::optional<Tensor>& preallocated_output) {
    using Op = MoeAgUntilizeActiveDeviceOperation;
    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{.experts_per_chip = experts_per_chip, .tiles_per_block = tiles_per_block},
        Op::tensor_args_t{
            .y = y,
            .counts = counts,
            .regions = regions,
            .local_slot_map = local_slot_map,
            .preallocated_output = preallocated_output});
}

// ---------------------------------------------------------------- untilize_x
void MoeAgUntilizeXDeviceOperation::validate_on_program_cache_miss(
    const MoeAgUntilizeXParams& args, const MoeAgUntilizeXInputs& t) {
    constexpr const char* op = "moe_ag_untilize_x";
    check_dram_interleaved(t.x, op, "x");
    TT_FATAL(t.x.layout() == Layout::TILE && t.x.dtype() == DataType::BFLOAT16, "{}: x must be bfloat16 TILE", op);
    const auto& s = t.x.logical_shape();
    TT_FATAL(s[-1] % 1024 == 0, "{}: hidden {} must be a multiple of 1024", op, s[-1]);
    TT_FATAL(s[-2] % 32 == 0, "{}: rows {} must be a multiple of 32", op, s[-2]);
    check_output(t.preallocated_output, compute_output_specs(args, t), op);
}

TensorSpec MoeAgUntilizeXDeviceOperation::compute_output_specs(
    const MoeAgUntilizeXParams&, const MoeAgUntilizeXInputs& t) {
    const auto& s = t.x.logical_shape();
    uint32_t rows = 1;
    for (int i = 0; i + 1 < static_cast<int>(s.rank()); ++i) {
        rows *= s[i];
    }
    return bf16_spec(ttnn::Shape({1, 1, rows * (s[-1] / 1024), 1024}), Layout::ROW_MAJOR);
}

Tensor MoeAgUntilizeXDeviceOperation::create_output_tensors(
    const MoeAgUntilizeXParams& args, const MoeAgUntilizeXInputs& t) {
    return output_or_new(t.preallocated_output, compute_output_specs(args, t), t.x);
}

Tensor moe_ag_untilize_x(const Tensor& x, const std::optional<Tensor>& preallocated_output) {
    using Op = MoeAgUntilizeXDeviceOperation;
    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{}, Op::tensor_args_t{.x = x, .preallocated_output = preallocated_output});
}

}  // namespace ttnn::prim
