// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flash_mqa_decode_device_operation.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::flash_mqa_decode {

using namespace tt::tt_metal;

FlashMqaDecodeDeviceOperation::program_factory_t FlashMqaDecodeDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ProgramFactory{};
}

void FlashMqaDecodeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& q = tensor_args.q;
    const auto& kv = tensor_args.kv;
    const auto& lengths = tensor_args.lengths;
    for (const Tensor* t : {&q, &kv, &lengths}) {
        TT_FATAL(t->storage_type() == StorageType::DEVICE, "flash_mqa_decode: inputs must be on device");
    }

    const auto& q_shape = q.logical_shape();
    TT_FATAL(
        q_shape.rank() == 4 && q_shape[0] == 1 && q_shape[1] == 1,
        "flash_mqa_decode: q must be [1, 1, heads, head_dim], got {}",
        q_shape);
    TT_FATAL(
        q.layout() == Layout::ROW_MAJOR && q.dtype() == DataType::BFLOAT16,
        "flash_mqa_decode: q must be row-major bfloat16");
    const uint32_t rows = args.num_rows;
    TT_FATAL(
        rows == 1 || rows == 2 || rows == 4 || rows == 8, "flash_mqa_decode: num_rows ({}) must be 1, 2, 4 or 8", rows);
    TT_FATAL(q_shape[2] % rows == 0, "flash_mqa_decode: q rows ({}) must be heads * num_rows ({})", q_shape[2], rows);
    const uint32_t heads = q_shape[2] / rows;
    const uint32_t head_dim = q_shape[3];
    TT_FATAL(heads % kHeadsPerCore == 0, "flash_mqa_decode: heads ({}) must be a multiple of {}", heads, kHeadsPerCore);
    // vDHt must split into 8-tile dst blocks for the tail reduction.
    TT_FATAL(head_dim % 256 == 0, "flash_mqa_decode: head_dim ({}) must be a multiple of 256", head_dim);

    TT_FATAL(
        args.k_chunk_size % 64 == 0 && args.k_chunk_size >= 64 && args.k_chunk_size <= 448,
        "flash_mqa_decode: k_chunk_size ({}) must be a multiple of 64 in [64, 448]",
        args.k_chunk_size);
    TT_FATAL(
        args.num_s_blocks >= 1 && args.num_s_blocks <= 32,
        "flash_mqa_decode: num_s_blocks ({}) must be in [1, 32]",
        args.num_s_blocks);

    const auto check_kv = [&](const Tensor& t, const char* name) {
        const auto& shape = t.logical_shape();
        TT_FATAL(
            shape.rank() == 4 && shape[0] == 1 && shape[1] == 1 && shape[3] == head_dim,
            "flash_mqa_decode: {} must be [1, 1, rows, {}], got {}",
            name,
            head_dim,
            shape);
        TT_FATAL(
            t.layout() == Layout::TILE && t.tensor_spec().tile().get_height() == 32 &&
                t.tensor_spec().tile().get_width() == 32,
            "flash_mqa_decode: {} must be in 32x32 TILE layout",
            name);
        TT_FATAL(
            t.dtype() == DataType::BFLOAT16 || t.dtype() == DataType::BFLOAT8_B,
            "flash_mqa_decode: {} must be bfloat16 or bfloat8_b, got {}",
            name,
            t.dtype());
    };
    check_kv(kv, "kv");
    TT_FATAL(
        kv.logical_shape()[2] % args.k_chunk_size == 0,
        "flash_mqa_decode: kv rows ({}) must be a multiple of k_chunk_size ({})",
        kv.logical_shape()[2],
        args.k_chunk_size);
    TT_FATAL(
        args.ring_rows % args.k_chunk_size == 0 && args.ring_rows <= kv.logical_shape()[2],
        "flash_mqa_decode: ring_rows ({}) must be a multiple of k_chunk_size ({}) within kv's {} rows",
        args.ring_rows,
        args.k_chunk_size,
        kv.logical_shape()[2]);
    if (rows > 1) {
        TT_FATAL(
            tensor_args.new_kv.has_value(), "flash_mqa_decode: num_rows > 1 needs new_kv (the block's new kv rows)");
        const auto& new_kv = tensor_args.new_kv.value();
        TT_FATAL(new_kv.storage_type() == StorageType::DEVICE, "flash_mqa_decode: new_kv must be on device");
        check_kv(new_kv, "new_kv");
        TT_FATAL(
            new_kv.logical_shape()[2] >= tt::constants::TILE_HEIGHT && new_kv.dtype() == kv.dtype(),
            "flash_mqa_decode: new_kv must have at least 32 rows and kv's dtype, got {} {}",
            new_kv.logical_shape(),
            new_kv.dtype());
        TT_FATAL(tensor_args.cos.has_value(), "flash_mqa_decode: num_rows > 1 needs cos / sin (per-row inverse RoPE)");
        TT_FATAL(
            (lengths.dtype() == DataType::INT32 || lengths.dtype() == DataType::UINT32) &&
                lengths.layout() == Layout::ROW_MAJOR && lengths.logical_shape()[-1] >= 4 + rows,
            "flash_mqa_decode: num_rows {} needs row-major int32 lengths [p, ring_first_row, selected_first_row, 0, "
            "n_0 .. n_{}], got {} {}",
            rows,
            rows - 1,
            lengths.dtype(),
            lengths.logical_shape());
    } else {
        TT_FATAL(!tensor_args.new_kv.has_value(), "flash_mqa_decode: new_kv needs num_rows > 1");
        TT_FATAL(!args.dense_selected, "flash_mqa_decode: dense_selected needs num_rows > 1");
    }
    const bool per_row_selected = rows > 1 && !args.dense_selected;
    if (tensor_args.selected_kv.has_value()) {
        const auto& selected = tensor_args.selected_kv.value();
        TT_FATAL(selected.storage_type() == StorageType::DEVICE, "flash_mqa_decode: selected_kv must be on device");
        check_kv(selected, "selected_kv");
        TT_FATAL(selected.dtype() == kv.dtype(), "flash_mqa_decode: selected_kv and kv must share a dtype");
        const uint32_t selected_rows = selected.logical_shape()[2];
        TT_FATAL(
            args.selected_row_offset % 32 == 0 && args.selected_row_offset <= selected_rows,
            "flash_mqa_decode: selected_row_offset ({}) must be a multiple of 32 within selected_kv's {} rows",
            args.selected_row_offset,
            selected_rows);
        const uint32_t blocks = per_row_selected ? rows : 1;
        const uint32_t capacity =
            args.selected_rows ? args.selected_rows : (selected_rows - args.selected_row_offset) / blocks;
        TT_FATAL(
            capacity % args.k_chunk_size == 0 && args.selected_row_offset + blocks * capacity <= selected_rows,
            "flash_mqa_decode: selected rows ({}) must be a multiple of k_chunk_size ({}) within selected_kv's {} rows "
            "past the offset {}",
            capacity,
            args.k_chunk_size,
            selected_rows,
            args.selected_row_offset);
    } else {
        TT_FATAL(
            args.selected_row_offset == 0 && args.selected_rows == 0,
            "flash_mqa_decode: selected_row_offset and selected_rows need a selected_kv");
    }

    TT_FATAL(
        (lengths.dtype() == DataType::INT32 || lengths.dtype() == DataType::UINT32) &&
            lengths.layout() == Layout::ROW_MAJOR && lengths.logical_shape()[-1] >= 2,
        "flash_mqa_decode: lengths must be row-major int32 with [ring_valid, selected_valid(, ring_first_row(, "
        "selected_first_row))] in its first row, got {} {}",
        lengths.dtype(),
        lengths.logical_shape());
    TT_FATAL(
        rows > 1 || lengths.logical_shape()[-1] < 4 || tensor_args.selected_kv.has_value(),
        "flash_mqa_decode: a selected_first_row in lengths needs a selected_kv");

    if (tensor_args.attention_sink.has_value()) {
        const auto& sink = tensor_args.attention_sink.value();
        TT_FATAL(sink.storage_type() == StorageType::DEVICE, "flash_mqa_decode: attention_sink must be on device");
        TT_FATAL(
            sink.layout() == Layout::TILE && sink.dtype() == DataType::BFLOAT16 &&
                sink.tensor_spec().tile().get_height() == 32 && sink.logical_shape()[-2] == heads,
            "flash_mqa_decode: attention_sink must be a bfloat16 32x32 TILE [{}, 32] (sink / scale in column 0), got "
            "{}",
            heads,
            sink.logical_shape());
    }

    TT_FATAL(
        tensor_args.cos.has_value() == tensor_args.sin.has_value(),
        "flash_mqa_decode: cos and sin must both be provided or both omitted");
    if (tensor_args.cos.has_value()) {
        TT_FATAL(
            q.device()->arch() == tt::ARCH::BLACKHOLE,
            "flash_mqa_decode: fused inverse RoPE is only supported on Blackhole");
        const auto& cos = tensor_args.cos.value();
        const auto& sin = tensor_args.sin.value();
        for (const auto& [t, name] : {std::pair{&cos, "cos"}, std::pair{&sin, "sin"}}) {
            TT_FATAL(t->storage_type() == StorageType::DEVICE, "flash_mqa_decode: {} must be on device", name);
            const uint32_t tile_h = t->tensor_spec().tile().get_height();
            TT_FATAL(
                t->layout() == Layout::TILE && t->dtype() == DataType::BFLOAT16 &&
                    (tile_h == 32 || (rows > 1 && tile_h == rows)) && t->tensor_spec().tile().get_width() == 32,
                "flash_mqa_decode: {} must be bfloat16 TILE, 32x32 (or ({}, 32) with num_rows {})",
                name,
                rows,
                rows);
            const auto& s = t->logical_shape();
            TT_FATAL(
                s.rank() == 4 && s[0] == 1 && s[1] == 1 && s[2] >= rows,
                "flash_mqa_decode: {} must be [1, 1, >=num_rows, rope_dim], got {}",
                name,
                s);
        }
        const uint32_t rope_dim = cos.logical_shape()[3];
        TT_FATAL(
            sin.logical_shape()[3] == rope_dim,
            "flash_mqa_decode: cos ({}) and sin ({}) must have the same rope_dim",
            rope_dim,
            sin.logical_shape()[3]);
        TT_FATAL(
            rope_dim > 0 && rope_dim % tt::constants::TILE_WIDTH == 0 && rope_dim <= 64 && rope_dim <= head_dim,
            "flash_mqa_decode: rope_dim ({}) must be a multiple of {} in [32, 64] and at most head_dim ({})",
            rope_dim,
            tt::constants::TILE_WIDTH,
            head_dim);
    }

    auto* device = q.device();
    const auto grid = device->compute_with_storage_grid_size();
    TT_FATAL(
        rows == 1 || (args.row_groups >= 1 && rows % args.row_groups == 0),
        "flash_mqa_decode: row_groups ({}) must divide num_rows ({})",
        args.row_groups,
        rows);
    TT_FATAL(
        args.num_s_blocks <= grid.x && heads / kHeadsPerCore * args.row_groups <= grid.y,
        "flash_mqa_decode: {} S-blocks x {} head groups x {} row groups do not fit the {}x{} grid",
        args.num_s_blocks,
        heads / kHeadsPerCore,
        args.row_groups,
        grid.x,
        grid.y);
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), args.compute_kernel_config);
    TT_FATAL(
        !fp32_dest_acc_en && !dst_full_sync_en,
        "flash_mqa_decode: fp32_dest_acc_en and dst_full_sync_en are not supported");
}

FlashMqaDecodeDeviceOperation::spec_return_value_t FlashMqaDecodeDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (args.num_rows > 1) {
        const auto& q_shape = tensor_args.q.logical_shape();
        const uint32_t width = q_shape[2] / args.num_rows * q_shape[3];
        return TensorSpec(
            ttnn::Shape({1, 1, args.num_rows, width}),
            TensorLayout(
                DataType::BFLOAT16,
                PageConfig(Layout::TILE, tt::tt_metal::Tile({args.num_rows, tt::constants::TILE_WIDTH})),
                args.output_mem_config));
    }
    return TensorSpec(
        tensor_args.q.logical_shape(),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), args.output_mem_config));
}

FlashMqaDecodeDeviceOperation::tensor_return_value_t FlashMqaDecodeDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.q.device());
}

}  // namespace ttnn::operations::experimental::deepseek::flash_mqa_decode

namespace ttnn::prim {

ttnn::Tensor flash_mqa_decode(
    const ttnn::Tensor& q,
    const ttnn::Tensor& kv,
    const ttnn::Tensor& lengths,
    const std::optional<ttnn::Tensor>& selected_kv,
    uint32_t selected_row_offset,
    uint32_t ring_rows,
    uint32_t selected_rows,
    const std::optional<ttnn::Tensor>& attention_sink,
    std::optional<float> scale,
    uint32_t k_chunk_size,
    uint32_t num_s_blocks,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& cos,
    const std::optional<ttnn::Tensor>& sin,
    const std::optional<ttnn::Tensor>& new_kv,
    uint32_t num_rows,
    bool dense_selected,
    uint32_t row_groups) {
    using OperationType = ttnn::operations::experimental::deepseek::flash_mqa_decode::FlashMqaDecodeDeviceOperation;
    auto kernel_config_val = init_device_compute_kernel_config(
        q.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false);
    tt::tt_metal::MemoryConfig output_mem_config = memory_config.value_or(q.memory_config());
    if (num_rows > 1 && !memory_config.has_value() && q.memory_config().is_sharded()) {
        // The block on one core: q's first shard core, as the one-row output sits on q's (only) core.
        const auto& q_grid = q.memory_config().shard_spec()->grid;
        const auto first = q_grid.ranges().front().start_coord;
        const uint32_t width = q.logical_shape()[2] / num_rows * q.logical_shape()[3];
        output_mem_config = tt::tt_metal::MemoryConfig(
            tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            tt::tt_metal::BufferType::L1,
            tt::tt_metal::ShardSpec(
                CoreRangeSet(CoreRange(first, first)), {num_rows, width}, tt::tt_metal::ShardOrientation::ROW_MAJOR));
    }
    if (num_rows > 1 && row_groups == 0) {
        // As many row groups as divide the rows and fit the grid's height under the head groups.
        const uint32_t grid_y = q.device()->compute_with_storage_grid_size().y;
        const uint32_t head_groups = std::max<uint32_t>(
            1,
            q.logical_shape()[2] / num_rows /
                ttnn::operations::experimental::deepseek::flash_mqa_decode::kHeadsPerCore);
        row_groups = 1;
        for (uint32_t rg = num_rows; rg >= 1; --rg) {
            if (num_rows % rg == 0 && head_groups * rg <= grid_y) {
                row_groups = rg;
                break;
            }
        }
    }
    auto attrs = OperationType::operation_attributes_t{
        .scale = scale.value_or(1.0f / std::sqrt(static_cast<float>(q.logical_shape()[-1]))),
        .k_chunk_size = k_chunk_size,
        .num_s_blocks = num_s_blocks,
        .selected_row_offset = selected_row_offset,
        .ring_rows = ring_rows,
        .selected_rows = selected_rows,
        .num_rows = num_rows,
        .dense_selected = dense_selected,
        .row_groups = num_rows > 1 ? row_groups : 1,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = kernel_config_val,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .q = q,
        .kv = kv,
        .lengths = lengths,
        .selected_kv = selected_kv,
        .attention_sink = attention_sink,
        .cos = cos,
        .sin = sin,
        .new_kv = new_kv,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
