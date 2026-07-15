// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "masked_per_token_cast_back_device_operation.hpp"

#include <cstdint>
#include <limits>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/per_token_cast_to_fp8.hpp"

namespace ttnn::experimental::prim::masked_per_token_cast_back {

namespace fp8 = ttnn::operations::experimental::deepseek_prefill::per_token_cast_to_fp8;

namespace {

bool is_dram_interleaved(const tt::tt_metal::MemoryConfig& mem_config) {
    return mem_config.buffer_type() == tt::tt_metal::BufferType::DRAM &&
           mem_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
}

void validate_tensor_specs(const Tensor& tensor, const std::string& name) {
    TT_FATAL(tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "{} must have a buffer", name);
    TT_FATAL(tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "{} must be ROW_MAJOR", name);
    TT_FATAL(is_dram_interleaved(tensor.memory_config()), "{} must be DRAM interleaved", name);
}

void validate_index_tensor(const Tensor& tensor, const std::string& name) {
    validate_tensor_specs(tensor, name);
    TT_FATAL(tensor.dtype() == tt::tt_metal::DataType::UINT32, "{} must be UINT32, got {}", name, tensor.dtype());
    const auto& shape = tensor.logical_shape();
    const auto rank = shape.rank();
    const bool valid_1d = rank == 1;
    const bool valid_2d = rank == 2 && shape[0] == 1;
    TT_FATAL(valid_1d || valid_2d, "{} must be 1D or 2D with first dimension == 1, got shape {}", name, shape);
}

}  // namespace

MaskedPerTokenCastBackDeviceOperation::program_factory_t MaskedPerTokenCastBackDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MaskedPerTokenCastBackProgramFactory{};
}

void MaskedPerTokenCastBackDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input_e4m3 = tensor_args.input_e4m3;
    const auto& input_scale = tensor_args.input_scale;
    const auto& expert_region_offsets = tensor_args.expert_region_offsets;
    const auto& expert_token_counts = tensor_args.expert_token_counts;
    const auto& global_expert_idx_table = tensor_args.global_expert_idx_table;

    const bool scales_from_metadata = attrs.scales_from_metadata;

    validate_tensor_specs(input_e4m3, "masked_per_token_cast_back: input_e4m3");
    validate_tensor_specs(input_scale, "masked_per_token_cast_back: input_scale");
    validate_index_tensor(expert_region_offsets, "masked_per_token_cast_back: expert_region_offsets");
    validate_index_tensor(expert_token_counts, "masked_per_token_cast_back: expert_token_counts");
    validate_index_tensor(global_expert_idx_table, "masked_per_token_cast_back: global_expert_idx_table");
    TT_FATAL(
        is_dram_interleaved(attrs.output_memory_config),
        "masked_per_token_cast_back: output memory config must be DRAM interleaved");
    TT_FATAL(
        input_e4m3.device() == input_scale.device(),
        "masked_per_token_cast_back: input_e4m3 and input_scale must be on the same device");
    TT_FATAL(
        input_e4m3.device()->arch() == tt::ARCH::BLACKHOLE,
        "masked_per_token_cast_back: FP8_E4M3 path requires Blackhole hardware, got arch {}",
        input_e4m3.device()->arch());
    TT_FATAL(
        input_e4m3.dtype() == tt::tt_metal::DataType::FP8_E4M3,
        "masked_per_token_cast_back: input_e4m3 dtype must be FP8_E4M3");
    // Scale source dtype: plain scale path accepts FLOAT32 or BFLOAT16; metadata path carries the fp32
    // scales bit-stored in the int32 metadata tail, so UINT32/INT32 are also accepted there.
    if (scales_from_metadata) {
        const auto sdt = input_scale.dtype();
        TT_FATAL(
            sdt == tt::tt_metal::DataType::FLOAT32 || sdt == tt::tt_metal::DataType::UINT32 ||
                sdt == tt::tt_metal::DataType::INT32,
            "masked_per_token_cast_back: metadata scale source dtype must be FLOAT32/UINT32/INT32, got {}",
            sdt);
    } else {
        const auto sdt = input_scale.dtype();
        TT_FATAL(
            sdt == tt::tt_metal::DataType::FLOAT32 || sdt == tt::tt_metal::DataType::BFLOAT16,
            "masked_per_token_cast_back: input_scale dtype must be FLOAT32 or BFLOAT16, got {}",
            sdt);
    }

    const auto tile_shape = input_e4m3.tensor_spec().tile().get_tile_shape();
    const uint32_t tile_h = tile_shape[0];
    const uint32_t tile_w = tile_shape[1];

    // Row-major circular-buffer pages still use one logical tile. The quantization kernels then stream
    // those pages as tile-height batches of 128-element scale blocks.
    constexpr uint32_t ROW_MAJOR_TILE_ELEMS = 1024;
    TT_FATAL(
        tile_h * tile_w == ROW_MAJOR_TILE_ELEMS,
        "masked_per_token_cast_back: tile_h * tile_w must equal ROW_MAJOR_TILE_ELEMS={} for row-major block "
        "tilization, got {}x{}",
        ROW_MAJOR_TILE_ELEMS,
        tile_h,
        tile_w);
    TT_FATAL(
        fp8::BLOCK_W % tile_w == 0,
        "masked_per_token_cast_back: tile width {} must divide BLOCK_W={}",
        tile_w,
        fp8::BLOCK_W);
    TT_FATAL(
        attrs.output_dtype == tt::tt_metal::DataType::BFLOAT16 || attrs.output_dtype == tt::tt_metal::DataType::FLOAT32,
        "masked_per_token_cast_back: output_dtype must be BFLOAT16 or FLOAT32");
    TT_FATAL(attrs.experts_per_chip > 0, "masked_per_token_cast_back: experts_per_chip must be > 0");
    TT_FATAL(
        attrs.experts_per_chip <= global_expert_idx_table.logical_shape()[-1],
        "masked_per_token_cast_back: experts_per_chip ({}) must be <= global_expert_idx_table last dim ({})",
        attrs.experts_per_chip,
        global_expert_idx_table.logical_shape()[-1]);

    const auto& e4m3_shape = input_e4m3.logical_shape();
    const auto& scale_shape = input_scale.logical_shape();
    const auto rank = e4m3_shape.size();
    TT_FATAL(rank >= 2, "masked_per_token_cast_back: input_e4m3 rank must be >= 2");
    TT_FATAL(
        e4m3_shape.size() == scale_shape.size(),
        "masked_per_token_cast_back: input_e4m3 ({}D) and input_scale ({}D) must have the same rank",
        e4m3_shape.size(),
        scale_shape.size());

    for (size_t i = 0; i + 1 < e4m3_shape.size(); ++i) {
        TT_FATAL(
            e4m3_shape[i] == scale_shape[i],
            "masked_per_token_cast_back: leading dim {} mismatch ({} vs {})",
            i,
            e4m3_shape[i],
            scale_shape[i]);
    }

    const uint32_t H = static_cast<uint32_t>(e4m3_shape[-1]);
    const uint32_t H_scale = static_cast<uint32_t>(scale_shape[-1]);
    if (scales_from_metadata) {
        // Metadata row = [routing fields ...][H/BLOCK_W fp32 scales]. Require H % BLOCK_W == 0 and that the
        // row is wide enough to hold the H/BLOCK_W scale tail after the leading routing columns.
        TT_FATAL(
            H % fp8::BLOCK_W == 0,
            "masked_per_token_cast_back: e4m3 last dim ({}) must be a multiple of BLOCK_W ({})",
            H,
            fp8::BLOCK_W);
        const uint32_t blocks_per_row = H / fp8::BLOCK_W;
        TT_FATAL(
            H_scale >= blocks_per_row,
            "masked_per_token_cast_back: metadata last dim ({}) must be >= H/BLOCK_W ({}) to hold the scale tail",
            H_scale,
            blocks_per_row);
    } else {
        // M and H are arbitrary (the kernels zero-pad the partial last tile-row / column-block). The
        // e4m3 width must equal scale_width * 128, which keeps H a multiple of the block width.
        TT_FATAL(
            H == H_scale * fp8::BLOCK_W,
            "masked_per_token_cast_back: e4m3 last dim ({}) must equal scale last dim ({}) * BLOCK_W ({})",
            H,
            H_scale,
            fp8::BLOCK_W);
    }

    uint64_t folded_M = 1;
    for (size_t i = 0; i + 1 < rank; ++i) {
        folded_M *= static_cast<uint64_t>(e4m3_shape[i]);
        TT_FATAL(
            folded_M <= std::numeric_limits<uint32_t>::max(),
            "masked_per_token_cast_back: folded row count M={} exceeds uint32_t range",
            folded_M);
    }
    TT_FATAL(
        static_cast<uint64_t>(e4m3_shape[rank - 1]) <= std::numeric_limits<uint32_t>::max(),
        "masked_per_token_cast_back: hidden dim H={} exceeds uint32_t range",
        e4m3_shape[rank - 1]);
    TT_FATAL(static_cast<uint32_t>(folded_M) > 0, "masked_per_token_cast_back: row count M must be > 0");
}

void MaskedPerTokenCastBackDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attrs, tensor_args);
}

MaskedPerTokenCastBackDeviceOperation::spec_return_value_t MaskedPerTokenCastBackDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return TensorSpec(
        tensor_args.input_e4m3.logical_shape(),
        tt::tt_metal::TensorLayout(
            attrs.output_dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR), attrs.output_memory_config));
}

MaskedPerTokenCastBackDeviceOperation::tensor_return_value_t
MaskedPerTokenCastBackDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input_e4m3.device());
}

ttsl::hash::hash_t MaskedPerTokenCastBackDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto tile_shape = tensor_args.input_e4m3.tensor_spec().tile().get_tile_shape();
    const auto face_shape = tensor_args.input_e4m3.tensor_spec().tile().get_face_shape();
    return tt::tt_metal::operation::hash_operation<MaskedPerTokenCastBackDeviceOperation>(
        attrs,
        tensor_args.input_e4m3.dtype(),
        // Scale dtype selects the compiled program (fp32/HiFi4 vs bf16/HiFi2 datapath, scale_elem_bytes),
        // so it must be part of the hash — otherwise a later call with a different scale dtype reuses the
        // wrong cached program.
        tensor_args.input_scale.dtype(),
        tensor_args.input_e4m3.memory_config(),
        tensor_args.input_scale.memory_config(),
        tensor_args.expert_region_offsets.memory_config(),
        tensor_args.expert_token_counts.memory_config(),
        tensor_args.global_expert_idx_table.memory_config(),
        tensor_args.input_e4m3.logical_shape(),
        tensor_args.input_scale.logical_shape(),
        tile_shape[0],
        tile_shape[1],
        face_shape[0],
        face_shape[1]);
}

}  // namespace ttnn::experimental::prim::masked_per_token_cast_back

namespace ttnn::prim {

ttnn::Tensor masked_per_token_cast_back(
    const Tensor& input_e4m3,
    const Tensor& input_scale,
    const Tensor& expert_region_offsets,
    const Tensor& expert_token_counts,
    const Tensor& global_expert_idx_table,
    uint32_t experts_per_chip,
    tt::tt_metal::DataType output_dtype,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    bool scales_from_metadata) {
    using OperationType = ttnn::experimental::prim::masked_per_token_cast_back::MaskedPerTokenCastBackDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        .output_dtype = output_dtype,
        .output_memory_config = output_memory_config,
        .experts_per_chip = experts_per_chip,
        .scales_from_metadata = scales_from_metadata};
    auto tensor_args = OperationType::tensor_args_t{
        .input_e4m3 = input_e4m3,
        .input_scale = input_scale,
        .expert_region_offsets = expert_region_offsets,
        .expert_token_counts = expert_token_counts,
        .global_expert_idx_table = global_expert_idx_table};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
