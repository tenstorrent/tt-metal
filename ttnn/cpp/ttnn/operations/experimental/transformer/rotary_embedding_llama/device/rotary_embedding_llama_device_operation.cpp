// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "rotary_embedding_llama_multi_core_program_factory.hpp"
#include "rotary_embedding_llama_sharded_program_factory.hpp"
#include "ttnn/device.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::transformer::rotary_embedding_llama {

RotaryEmbeddingLlamaDeviceOperation::program_factory_t RotaryEmbeddingLlamaDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    if (operation_attributes.is_decode_mode) {
        return program::RotaryEmbeddingLlamaMultiCoreSharded{};
    }
    return program::RotaryEmbeddingLlamaMultiCore{};
}

void RotaryEmbeddingLlamaDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void RotaryEmbeddingLlamaDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& cos = tensor_args.cos_cache;
    const auto& sin = tensor_args.sin_cache;
    const auto& trans_mat = tensor_args.trans_mat;

    auto* ref_device = input_tensor.device();
    // Validate inputs are on device and same device
    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE, "input tensor to rotary embedding need to be on device!");
    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "input tensor to rotary embedding need to be allocated in buffers on device!");
    TT_FATAL((input_tensor.layout() == Layout::TILE), "input tensor to rotary embedding must be tilized");

    TT_FATAL(cos.storage_type() == StorageType::DEVICE, "cos tensor to rotary embedding need to be on device!");
    TT_FATAL(cos.buffer() != nullptr, "cos tensor to rotary embedding need to be allocated in buffers on device!");
    TT_FATAL(cos.device() == ref_device, "cos tensor to rotary embedding need to be on same device!");
    TT_FATAL((cos.layout() == Layout::TILE), "cos tensor to rotary embedding must be tilized");

    TT_FATAL(sin.storage_type() == StorageType::DEVICE, "sin tensor to rotary embedding need to be on device!");
    TT_FATAL(sin.buffer() != nullptr, "sin tensor to rotary embedding need to be allocated in buffers on device!");
    TT_FATAL(sin.device() == ref_device, "sin tensor to rotary embedding need to be on same device!");
    TT_FATAL((sin.layout() == Layout::TILE), "sin tensor to rotary embedding must be tilized");

    TT_FATAL(
        trans_mat.storage_type() == StorageType::DEVICE,
        "transformation matrix to rotary embedding need to be on device!");
    TT_FATAL(
        trans_mat.buffer() != nullptr,
        "transformation matrix to rotary embedding need to be allocated in buffers on device!");
    TT_FATAL(trans_mat.device() == ref_device, "transformation matrix to rotary embedding need to be on same device!");
    TT_FATAL((trans_mat.layout() == Layout::TILE), "transformation matrix to rotary embedding must be tilized");

    uint32_t head_dim = input_tensor.logical_shape()[-1];
    TT_FATAL(
        head_dim <= 128 ||
            std::get<ttnn::WormholeComputeKernelConfig>(operation_attributes.compute_kernel_config).fp32_dest_acc_en ==
                false,
        "If head_dim is > 128, fp32_dest_acc_en must be False");
    // Check that head_dim is less than 256
    TT_FATAL(head_dim <= 256, "Head dim must be less than 256");
    // Check that head_dim is a multiple of 32
    TT_FATAL(head_dim % TILE_WIDTH == 0, "Head dim must be a multiple of TILE_WIDTH");

    TT_FATAL(
        input_tensor.dtype() == cos.dtype() && cos.dtype() == sin.dtype() && sin.dtype() == trans_mat.dtype() &&
            trans_mat.dtype() == DataType::BFLOAT16,
        "All input tensors must have dtype = bfloat16");
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == operation_attributes.output_mem_config.memory_layout(),
        "Input tensor and output tensor must have same memory layout");

    // Check that cos and sin have same dims
    TT_FATAL(cos.logical_shape() == sin.logical_shape(), "Cos and Sin dims must match");

    if (operation_attributes.is_decode_mode) {  // Decode mode validation
        uint32_t seq_len = input_tensor.logical_shape()[0];
        TT_FATAL(
            seq_len == 1,
            "rotary_embedding_llama currently only supports sharded inputs in decode mode, and therefore, seq_len (in "
            "dim 0) must be 1.");

        TT_FATAL(
            (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED),
            "Sharded inputs for RoPE must be HEIGHT_SHARDED.");
        TT_FATAL(
            (cos.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED),
            "cos tensor for RoPE must be HEIGHT_SHARDED.");
        TT_FATAL(
            (sin.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED),
            "sin tensor for RoPE must be HEIGHT_SHARDED.");
        TT_FATAL(
            (trans_mat.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED),
            "transformation matrix for RoPE must be HEIGHT_SHARDED.");

        uint32_t num_cores = input_tensor.device()->compute_with_storage_grid_size().x *
                             input_tensor.device()->compute_with_storage_grid_size().y;
        uint32_t batch = input_tensor.logical_shape()[1];
        TT_FATAL(
            batch <= num_cores,
            "In decode mode, RoPE is parallelized over batch dimension, and therefore, batch size must be less than or "
            "equal to the number of cores");

        // Checks for cos and sin
        TT_FATAL(batch == cos.logical_shape()[1], "Cos and Sin must have the same batch size as the input");
        // TODO: might be supported by kernel currently, but need to check with pytest
        TT_FATAL(
            cos.shard_spec()->shape[0] == TILE_HEIGHT,
            "In decode mode, RoPE only supports n_heads (shard_shape[0]) less than equal to TILE_HEIGHT");

        // Checks for transformation matrix
        TT_FATAL(
            trans_mat.logical_shape()[0] == 1 && trans_mat.logical_shape()[1] == 1,
            "Transformation matrix must have 1st & 2nd dim equal to 1");
        TT_FATAL(
            trans_mat.shard_spec()->shape[0] == TILE_HEIGHT,
            "Transformation matrix must have 3rd dim equal to TILE_HEIGHT");
        TT_FATAL(
            trans_mat.shard_spec()->shape[1] == TILE_WIDTH,
            "Transformation matrix must have 4rd dim equal to TILE_WIDTH");
    } else {  // Prefill mode validation
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Input tensor must be interleaved in prefill mode");

        // Checks for cos and sin
        TT_FATAL(
            cos.logical_shape()[0] == 1 && cos.logical_shape()[-1] == head_dim,
            "Cos dims must match input dims: cos.shape = {}, head_dim = {}",
            cos.logical_shape(),
            head_dim);
        // Check num_heads in cos/sin
        TT_FATAL(
            cos.logical_shape()[1] == input_tensor.logical_shape()[1] || cos.logical_shape()[1] == 1,
            "Num heads in cos/sin must match input tensor num heads or be 1. Expected {}, got {}",
            input_tensor.logical_shape()[1],
            cos.logical_shape()[1]);
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == sin.memory_config().memory_layout(),
            "Input tensor and sin tensor must have same memory layout");
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == cos.memory_config().memory_layout(),
            "Input tensor and cos tensor must have same memory layout");

        // Checks for transformation matrix
        TT_FATAL(
            trans_mat.logical_shape()[0] == 1 && trans_mat.logical_shape()[1] == 1,
            "Transformation matrix must have 1st & 2nd dim equal to 1");
        TT_FATAL(
            trans_mat.logical_shape()[-2] == TILE_HEIGHT,
            "Transformation matrix must have 3rd dim equal to TILE_HEIGHT");
        TT_FATAL(
            trans_mat.logical_shape()[-1] == TILE_WIDTH, "Transformation matrix must have 4rd dim equal to TILE_WIDTH");
    }
}

RotaryEmbeddingLlamaDeviceOperation::spec_return_value_t RotaryEmbeddingLlamaDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& shape = input_tensor.logical_shape();
    return {TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(),
            tt::tt_metal::PageConfig(input_tensor.layout()),
            operation_attributes.output_mem_config))};
}

RotaryEmbeddingLlamaDeviceOperation::tensor_return_value_t RotaryEmbeddingLlamaDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args)[0], tensor_args.input_tensor.device());
}

tt::stl::hash::hash_t RotaryEmbeddingLlamaDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<RotaryEmbeddingLlamaDeviceOperation>(
        operation_attributes, tensor_args);
}

}  // namespace ttnn::operations::experimental::transformer::rotary_embedding_llama

namespace ttnn::prim {

tt::tt_metal::Tensor rotary_embedding_llama(
    const tt::tt_metal::Tensor& input_tensor,
    const tt::tt_metal::Tensor& cos_cache,
    const tt::tt_metal::Tensor& sin_cache,
    const tt::tt_metal::Tensor& trans_mat,
    bool is_decode_mode,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType =
        ttnn::operations::experimental::transformer::rotary_embedding_llama::RotaryEmbeddingLlamaDeviceOperation;

    auto arch = input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch()
                                                                   : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);

    tt::tt_metal::MemoryConfig default_memory_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
    if (input_tensor.storage_type() == StorageType::DEVICE) {
        default_memory_config = input_tensor.memory_config();
    }

    auto operation_attributes = OperationType::operation_attributes_t{
        .is_decode_mode = is_decode_mode,
        .output_mem_config = memory_config.value_or(default_memory_config),
        .compute_kernel_config = kernel_config_val};
    auto tensor_args = OperationType::tensor_args_t{
        .input_tensor = input_tensor, .cos_cache = cos_cache, .sin_cache = sin_cache, .trans_mat = trans_mat};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
