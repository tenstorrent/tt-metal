// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flash_mqa_decode_device_operation.hpp"

#include <cmath>

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
    const uint32_t heads = q_shape[2];
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
    if (tensor_args.selected_kv.has_value()) {
        const auto& selected = tensor_args.selected_kv.value();
        TT_FATAL(selected.storage_type() == StorageType::DEVICE, "flash_mqa_decode: selected_kv must be on device");
        check_kv(selected, "selected_kv");
        TT_FATAL(selected.dtype() == kv.dtype(), "flash_mqa_decode: selected_kv and kv must share a dtype");
        const uint32_t rows = selected.logical_shape()[2];
        TT_FATAL(
            args.selected_row_offset % 32 == 0 && args.selected_row_offset <= rows,
            "flash_mqa_decode: selected_row_offset ({}) must be a multiple of 32 within selected_kv's {} rows",
            args.selected_row_offset,
            rows);
        TT_FATAL(
            (rows - args.selected_row_offset) % args.k_chunk_size == 0,
            "flash_mqa_decode: selected rows past the offset ({}) must be a multiple of k_chunk_size ({})",
            rows - args.selected_row_offset,
            args.k_chunk_size);
    } else {
        TT_FATAL(args.selected_row_offset == 0, "flash_mqa_decode: selected_row_offset needs a selected_kv");
    }

    TT_FATAL(
        (lengths.dtype() == DataType::INT32 || lengths.dtype() == DataType::UINT32) &&
            lengths.layout() == Layout::ROW_MAJOR && lengths.logical_shape()[-1] >= 2,
        "flash_mqa_decode: lengths must be row-major int32 with [ring_valid, selected_valid(, ring_first_row)] in "
        "its first row, got {} {}",
        lengths.dtype(),
        lengths.logical_shape());

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

    auto* device = q.device();
    const auto grid = device->compute_with_storage_grid_size();
    TT_FATAL(
        args.num_s_blocks <= grid.x && heads / kHeadsPerCore <= grid.y,
        "flash_mqa_decode: {} S-blocks x {} head groups do not fit the {}x{} grid",
        args.num_s_blocks,
        heads / kHeadsPerCore,
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
    const std::optional<ttnn::Tensor>& attention_sink,
    std::optional<float> scale,
    uint32_t k_chunk_size,
    uint32_t num_s_blocks,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::experimental::deepseek::flash_mqa_decode::FlashMqaDecodeDeviceOperation;
    auto kernel_config_val = init_device_compute_kernel_config(
        q.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false);
    auto attrs = OperationType::operation_attributes_t{
        .scale = scale.value_or(1.0f / std::sqrt(static_cast<float>(q.logical_shape()[-1]))),
        .k_chunk_size = k_chunk_size,
        .num_s_blocks = num_s_blocks,
        .selected_row_offset = selected_row_offset,
        .ring_rows = ring_rows,
        .output_mem_config = memory_config.value_or(q.memory_config()),
        .compute_kernel_config = kernel_config_val,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .q = q,
        .kv = kv,
        .lengths = lengths,
        .selected_kv = selected_kv,
        .attention_sink = attention_sink,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
