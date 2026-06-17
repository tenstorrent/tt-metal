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
    TT_FATAL(tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "{} must have a buffer", name);
    TT_FATAL(tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "{} must be ROW_MAJOR", name);
    TT_FATAL(is_dram_interleaved(tensor.memory_config()), "{} must be DRAM interleaved", name);
}

}  // namespace

PerTokenCastBackDeviceOperation::program_factory_t PerTokenCastBackDeviceOperation::select_program_factory(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    if (attrs.masked) {
        return MaskedPerTokenCastBackProgramFactory{};
    }
    return PerTokenCastBackProgramFactory{};
}

void PerTokenCastBackDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input_e4m3 = tensor_args.input_e4m3;
    const auto& input_scale = tensor_args.input_scale;

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
    TT_FATAL(
        input_scale.dtype() == tt::tt_metal::DataType::FLOAT32,
        "per_token_cast_back: input_scale dtype must be FLOAT32");
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

    // In the plain path each e4m3 row r uses scale row r, so the two tensors must agree on rank and
    // all leading (row) dims. In masked mode the scale is gathered by metadata token_idx and lives in
    // the original-token space (independent of the dispatch buffer's token dim), so that coupling does
    // not apply — only the per-128-block width relation below is enforced.
    if (!attrs.masked) {
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
    }

    const uint32_t H = static_cast<uint32_t>(e4m3_shape[-1]);
    const uint32_t H_scale = static_cast<uint32_t>(scale_shape[-1]);
    // M and H are arbitrary (the kernels zero-pad the partial last tile-row / column-block). The
    // e4m3 width must equal scale_width * 128, which keeps H a multiple of the block width.
    TT_FATAL(
        H == H_scale * fp8::BLOCK_W,
        "per_token_cast_back: e4m3 last dim ({}) must equal scale last dim ({}) * BLOCK_W ({})",
        H,
        H_scale,
        fp8::BLOCK_W);

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

    if (!attrs.masked) {
        return;
    }

    // ---- Masked decompress mode validation ----
    TT_FATAL(
        tensor_args.expert_token_counts.has_value() && tensor_args.expert_region_offsets.has_value() &&
            tensor_args.metadata.has_value(),
        "per_token_cast_back: masked mode requires expert_token_counts, expert_region_offsets, and metadata");
    TT_FATAL(
        attrs.experts_per_chip > 0 && attrs.dispatch_group_size > 0,
        "per_token_cast_back: masked mode requires experts_per_chip>0 ({}) and dispatch_group_size>0 ({})",
        attrs.experts_per_chip,
        attrs.dispatch_group_size);
    // The scale tensor is the dispatch group's gathered per-token scale: its token dim must split
    // evenly into the dispatch_group_size source slices (ISL = dispatch_group_size * seq_len_per_chip).
    const uint32_t scale_rows = static_cast<uint32_t>(scale_shape[scale_shape.size() - 2]);
    TT_FATAL(
        scale_rows % attrs.dispatch_group_size == 0,
        "per_token_cast_back: input_scale token dim ({}) must be divisible by dispatch_group_size ({})",
        scale_rows,
        attrs.dispatch_group_size);

    const auto& counts = *tensor_args.expert_token_counts;
    const auto& offsets = *tensor_args.expert_region_offsets;
    const auto& meta = *tensor_args.metadata;

    validate_tensor_specs(counts, "per_token_cast_back: expert_token_counts");
    validate_tensor_specs(offsets, "per_token_cast_back: expert_region_offsets");
    validate_tensor_specs(meta, "per_token_cast_back: metadata");

    TT_FATAL(
        counts.dtype() == tt::tt_metal::DataType::INT32 || counts.dtype() == tt::tt_metal::DataType::UINT32,
        "per_token_cast_back: expert_token_counts must be INT32 or UINT32, got {}",
        counts.dtype());
    TT_FATAL(
        offsets.dtype() == tt::tt_metal::DataType::INT32 || offsets.dtype() == tt::tt_metal::DataType::UINT32,
        "per_token_cast_back: expert_region_offsets must be INT32 or UINT32, got {}",
        offsets.dtype());
    TT_FATAL(
        meta.dtype() == tt::tt_metal::DataType::INT32,
        "per_token_cast_back: metadata must be INT32, got {}",
        meta.dtype());
    TT_FATAL(
        counts.device() == input_e4m3.device() && offsets.device() == input_e4m3.device() &&
            meta.device() == input_e4m3.device(),
        "per_token_cast_back: masked tensors must be on the same device as input_e4m3");

    TT_FATAL(
        offsets.logical_shape() == counts.logical_shape(),
        "per_token_cast_back: expert_region_offsets shape {} must match expert_token_counts shape {}",
        offsets.logical_shape(),
        counts.logical_shape());

    const uint32_t num_routed_experts = static_cast<uint32_t>(counts.logical_shape()[-1]);
    TT_FATAL(
        num_routed_experts % attrs.experts_per_chip == 0,
        "per_token_cast_back: num_routed_experts ({}) must be divisible by experts_per_chip ({})",
        num_routed_experts,
        attrs.experts_per_chip);
    const uint32_t group_experts = attrs.experts_per_chip * attrs.dispatch_group_size;
    TT_FATAL(
        group_experts > 0 && num_routed_experts % group_experts == 0,
        "per_token_cast_back: num_routed_experts ({}) must be divisible by experts_per_chip*dispatch_group_size ({})",
        num_routed_experts,
        group_experts);

    // input_e4m3 is the dispatch buffer (1, 1, T, H); metadata is (1, 1, T, 5). T must match.
    const auto& meta_shape = meta.logical_shape();
    TT_FATAL(
        meta_shape.size() == 4,
        "per_token_cast_back: metadata must be rank-4 (1,1,T,5), got rank {}",
        meta_shape.size());
    TT_FATAL(meta_shape[-1] == 5, "per_token_cast_back: metadata last dim must be 5, got {}", meta_shape[-1]);
    TT_FATAL(
        static_cast<uint32_t>(e4m3_shape[-2]) == static_cast<uint32_t>(meta_shape[-2]),
        "per_token_cast_back: input_e4m3 token dim ({}) must match metadata token dim ({})",
        e4m3_shape[-2],
        meta_shape[-2]);
}

void PerTokenCastBackDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attrs, tensor_args);
}

PerTokenCastBackDeviceOperation::spec_return_value_t PerTokenCastBackDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return TensorSpec(
        tensor_args.input_e4m3.logical_shape(),
        tt::tt_metal::TensorLayout(
            attrs.output_dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR), attrs.output_memory_config));
}

PerTokenCastBackDeviceOperation::tensor_return_value_t PerTokenCastBackDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input_e4m3.device());
}

tt::stl::hash::hash_t PerTokenCastBackDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto tile_shape = tensor_args.input_e4m3.tensor_spec().tile().get_tile_shape();
    const auto face_shape = tensor_args.input_e4m3.tensor_spec().tile().get_face_shape();
    // The masked flag and its layout params select a different program-factory variant (and entirely
    // different kernels), so they MUST be part of the hash — otherwise a cached plain-mode program
    // could be wrongly reused for masked inputs. When masked, the routing-tensor shapes also matter.
    const auto masked_meta_shape =
        tensor_args.metadata.has_value() ? tensor_args.metadata->logical_shape() : ttnn::Shape{};
    const auto masked_counts_shape =
        tensor_args.expert_token_counts.has_value() ? tensor_args.expert_token_counts->logical_shape() : ttnn::Shape{};
    return tt::tt_metal::operation::hash_operation<PerTokenCastBackDeviceOperation>(
        attrs,
        attrs.masked,
        attrs.experts_per_chip,
        attrs.dispatch_group_size,
        tensor_args.input_e4m3.dtype(),
        tensor_args.input_e4m3.memory_config(),
        tensor_args.input_scale.memory_config(),
        tensor_args.input_e4m3.logical_shape(),
        tensor_args.input_scale.logical_shape(),
        masked_meta_shape,
        masked_counts_shape,
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
    const std::optional<Tensor>& expert_token_counts,
    const std::optional<Tensor>& expert_region_offsets,
    const std::optional<Tensor>& metadata,
    uint32_t experts_per_chip,
    uint32_t dispatch_group_size) {
    using OperationType = ttnn::experimental::prim::per_token_cast_back::PerTokenCastBackDeviceOperation;
    const bool masked = expert_token_counts.has_value();
    auto operation_attributes = OperationType::operation_attributes_t{
        .output_dtype = output_dtype,
        .output_memory_config = output_memory_config,
        .masked = masked,
        .experts_per_chip = experts_per_chip,
        .dispatch_group_size = dispatch_group_size};
    auto tensor_args = OperationType::tensor_args_t{
        .input_e4m3 = input_e4m3,
        .input_scale = input_scale,
        .expert_token_counts = expert_token_counts,
        .expert_region_offsets = expert_region_offsets,
        .metadata = metadata};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
