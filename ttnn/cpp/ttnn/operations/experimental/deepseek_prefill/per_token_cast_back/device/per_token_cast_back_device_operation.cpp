// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "per_token_cast_back_device_operation.hpp"

#include <cstdint>
#include <limits>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/per_token_cast_to_fp8.hpp"

namespace ttnn::experimental::prim::per_token_cast_back {

namespace fp8 = ttnn::operations::experimental::deepseek_prefill::per_token_cast_to_fp8;

namespace {

bool is_dram_interleaved(const tt::tt_metal::MemoryConfig& mem_config) {
    return mem_config.buffer_type() == tt::tt_metal::BufferType::DRAM &&
           mem_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
}

void validate_tensor_specs(const Tensor& tensor, const std::string& name) {
    TT_FATAL(tensor.storage_type() == ttnn::StorageType::DEVICE, "{} must be on device", name);
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

PerTokenCastBackDeviceOperation::program_factory_t PerTokenCastBackDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return PerTokenCastBackProgramFactory{};
}

void PerTokenCastBackDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input_e4m3 = tensor_args.input_e4m3;
    const auto& input_scale = tensor_args.input_scale;
    const bool token_count_aware = attrs.token_count_aware;
    const bool scales_from_metadata = attrs.scales_from_metadata;

    validate_tensor_specs(input_e4m3, "per_token_cast_back: input_e4m3");
    validate_tensor_specs(input_scale, "per_token_cast_back: input_scale");
    TT_FATAL(
        is_dram_interleaved(attrs.output_memory_config),
        "per_token_cast_back: output memory config must be DRAM interleaved");
    TT_FATAL(
        input_e4m3.device() == input_scale.device(),
        "per_token_cast_back: input_e4m3 and input_scale must be on the same device");
    TT_FATAL(
        input_e4m3.device()->arch() == tt::ARCH::BLACKHOLE,
        "per_token_cast_back: FP8_E4M3 path requires Blackhole hardware, got arch {}",
        input_e4m3.device()->arch());
    TT_FATAL(
        input_e4m3.dtype() == tt::tt_metal::DataType::FP8_E4M3,
        "per_token_cast_back: input_e4m3 dtype must be FP8_E4M3");

    if (token_count_aware) {
        const auto& expert_region_offsets = tensor_args.expert_region_offsets;
        const auto& expert_token_counts = tensor_args.expert_token_counts;
        const auto& global_expert_idx_table = tensor_args.global_expert_idx_table;
        TT_FATAL(
            expert_region_offsets.has_value() && expert_token_counts.has_value() && global_expert_idx_table.has_value(),
            "per_token_cast_back: token_count_aware path requires expert_region_offsets, expert_token_counts and "
            "global_expert_idx_table");
        validate_index_tensor(*expert_region_offsets, "per_token_cast_back: expert_region_offsets");
        validate_index_tensor(*expert_token_counts, "per_token_cast_back: expert_token_counts");
        validate_index_tensor(*global_expert_idx_table, "per_token_cast_back: global_expert_idx_table");
        TT_FATAL(attrs.experts_per_chip > 0, "per_token_cast_back: experts_per_chip must be > 0");
        TT_FATAL(
            attrs.experts_per_chip <= global_expert_idx_table->logical_shape()[-1],
            "per_token_cast_back: experts_per_chip ({}) must be <= global_expert_idx_table last dim ({})",
            attrs.experts_per_chip,
            global_expert_idx_table->logical_shape()[-1]);
        // Scale source dtype: scales are always fp32. The plain scale path requires FLOAT32; the metadata
        // path carries the fp32 scales bit-stored in the int32 metadata tail, so UINT32/INT32 are accepted there.
        if (scales_from_metadata) {
            const auto sdt = input_scale.dtype();
            TT_FATAL(
                sdt == tt::tt_metal::DataType::FLOAT32 || sdt == tt::tt_metal::DataType::UINT32 ||
                    sdt == tt::tt_metal::DataType::INT32,
                "per_token_cast_back: metadata scale source dtype must be FLOAT32/UINT32/INT32, got {}",
                sdt);
        } else {
            TT_FATAL(
                input_scale.dtype() == tt::tt_metal::DataType::FLOAT32,
                "per_token_cast_back: input_scale dtype must be FLOAT32, got {}",
                input_scale.dtype());
        }
    } else {
        TT_FATAL(
            !attrs.scales_from_metadata,
            "per_token_cast_back: scales_from_metadata is only valid on the token_count_aware path");
        TT_FATAL(
            input_scale.dtype() == tt::tt_metal::DataType::FLOAT32,
            "per_token_cast_back: input_scale dtype must be FLOAT32");
    }

    const auto tile_shape = input_e4m3.tensor_spec().tile().get_tile_shape();
    const uint32_t tile_h = tile_shape[0];
    const uint32_t tile_w = tile_shape[1];

    // Row-major circular-buffer pages still use one logical tile. The quantization kernels then stream
    // those pages as tile-height batches of 128-element scale blocks.
    constexpr uint32_t ROW_MAJOR_TILE_ELEMS = 1024;
    TT_FATAL(
        tile_h * tile_w == ROW_MAJOR_TILE_ELEMS,
        "per_token_cast_back: tile_h * tile_w must equal ROW_MAJOR_TILE_ELEMS={} for row-major block tilization, "
        "got {}x{}",
        ROW_MAJOR_TILE_ELEMS,
        tile_h,
        tile_w);
    TT_FATAL(
        fp8::BLOCK_W % tile_w == 0, "per_token_cast_back: tile width {} must divide BLOCK_W={}", tile_w, fp8::BLOCK_W);
    TT_FATAL(
        attrs.output_dtype == tt::tt_metal::DataType::BFLOAT16 || attrs.output_dtype == tt::tt_metal::DataType::FLOAT32,
        "per_token_cast_back: output_dtype must be BFLOAT16 or FLOAT32");

    const auto& e4m3_shape = input_e4m3.logical_shape();
    const auto& scale_shape = input_scale.logical_shape();
    const auto rank = e4m3_shape.size();
    TT_FATAL(rank >= 2, "per_token_cast_back: input_e4m3 rank must be >= 2");
    TT_FATAL(
        e4m3_shape.size() == scale_shape.size(),
        "per_token_cast_back: input_e4m3 ({}D) and input_scale ({}D) must have the same rank",
        e4m3_shape.size(),
        scale_shape.size());

    for (size_t i = 0; i + 1 < e4m3_shape.size(); ++i) {
        TT_FATAL(
            e4m3_shape[i] == scale_shape[i],
            "per_token_cast_back: leading dim {} mismatch ({} vs {})",
            i,
            e4m3_shape[i],
            scale_shape[i]);
    }

    const uint32_t H = static_cast<uint32_t>(e4m3_shape[-1]);
    const uint32_t H_scale = static_cast<uint32_t>(scale_shape[-1]);
    if (token_count_aware && scales_from_metadata) {
        // Metadata row = [routing fields ...][H/BLOCK_W fp32 scales]. Require H % BLOCK_W == 0 and that the
        // row is wide enough to hold the H/BLOCK_W scale tail after the leading routing columns.
        TT_FATAL(
            H % fp8::BLOCK_W == 0,
            "per_token_cast_back: e4m3 last dim ({}) must be a multiple of BLOCK_W ({})",
            H,
            fp8::BLOCK_W);
        const uint32_t blocks_per_row = H / fp8::BLOCK_W;
        TT_FATAL(
            H_scale >= blocks_per_row,
            "per_token_cast_back: metadata last dim ({}) must be >= H/BLOCK_W ({}) to hold the scale tail",
            H_scale,
            blocks_per_row);
    } else {
        // M and H are arbitrary (the kernels zero-pad the partial last tile-row / column-block). The
        // e4m3 width must equal scale_width * 128, which keeps H a multiple of the block width.
        TT_FATAL(
            H == H_scale * fp8::BLOCK_W,
            "per_token_cast_back: e4m3 last dim ({}) must equal scale last dim ({}) * BLOCK_W ({})",
            H,
            H_scale,
            fp8::BLOCK_W);
    }

    uint64_t folded_M = 1;
    for (size_t i = 0; i + 1 < rank; ++i) {
        folded_M *= static_cast<uint64_t>(e4m3_shape[i]);
        TT_FATAL(
            folded_M <= std::numeric_limits<uint32_t>::max(),
            "per_token_cast_back: folded row count M={} exceeds uint32_t range",
            folded_M);
    }
    TT_FATAL(
        static_cast<uint64_t>(e4m3_shape[rank - 1]) <= std::numeric_limits<uint32_t>::max(),
        "per_token_cast_back: hidden dim H={} exceeds uint32_t range",
        e4m3_shape[rank - 1]);
    TT_FATAL(static_cast<uint32_t>(folded_M) > 0, "per_token_cast_back: row count M must be > 0");
}

void PerTokenCastBackDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attrs, tensor_args);
}

PerTokenCastBackDeviceOperation::spec_return_value_t PerTokenCastBackDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return tt::tt_metal::TensorSpec(
        tensor_args.input_e4m3.logical_shape(),
        tt::tt_metal::TensorLayout(
            attrs.output_dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR), attrs.output_memory_config));
}

PerTokenCastBackDeviceOperation::tensor_return_value_t PerTokenCastBackDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input_e4m3.device());
}

ttsl::hash::hash_t PerTokenCastBackDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto tile_shape = tensor_args.input_e4m3.tensor_spec().tile().get_tile_shape();
    const auto face_shape = tensor_args.input_e4m3.tensor_spec().tile().get_face_shape();
    // Token-count-aware metadata tensors, when present, contribute their memory configs. Their absence
    // (plain path) is already distinguished by attrs.token_count_aware, which hash_operation folds in via `attrs`.
    const auto opt_mem_config = [](const std::optional<Tensor>& t) {
        return t.has_value() ? std::optional<tt::tt_metal::MemoryConfig>(t->memory_config()) : std::nullopt;
    };
    return tt::tt_metal::operation::hash_operation<PerTokenCastBackDeviceOperation>(
        attrs,
        tensor_args.input_e4m3.dtype(),
        // Scale source dtype (fp32 vs int32/uint32 metadata) affects the reader.
        tensor_args.input_scale.dtype(),
        tensor_args.input_e4m3.memory_config(),
        tensor_args.input_scale.memory_config(),
        opt_mem_config(tensor_args.expert_region_offsets),
        opt_mem_config(tensor_args.expert_token_counts),
        opt_mem_config(tensor_args.global_expert_idx_table),
        tensor_args.input_e4m3.logical_shape(),
        tensor_args.input_scale.logical_shape(),
        tile_shape[0],
        tile_shape[1],
        face_shape[0],
        face_shape[1]);
}

}  // namespace ttnn::experimental::prim::per_token_cast_back

namespace ttnn::prim {

ttnn::Tensor per_token_cast_back(
    const Tensor& input_e4m3,
    const Tensor& input_scale,
    tt::tt_metal::DataType output_dtype,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    bool token_count_aware,
    const std::optional<Tensor>& expert_region_offsets,
    const std::optional<Tensor>& expert_token_counts,
    const std::optional<Tensor>& global_expert_idx_table,
    uint32_t experts_per_chip,
    bool scales_from_metadata) {
    using OperationType = ttnn::experimental::prim::per_token_cast_back::PerTokenCastBackDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        .output_dtype = output_dtype,
        .output_memory_config = output_memory_config,
        .token_count_aware = token_count_aware,
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
