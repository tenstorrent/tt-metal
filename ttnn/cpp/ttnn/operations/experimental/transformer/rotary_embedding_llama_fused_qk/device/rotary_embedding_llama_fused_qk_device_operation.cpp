// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/device/rotary_embedding_llama_fused_qk_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::operations::experimental::transformer::rotary_embedding_llama_fused_qk {

RotaryEmbeddingLlamaFusedQKDeviceOperation::program_factory_t
RotaryEmbeddingLlamaFusedQKDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return program::RotaryEmbeddingLlamaFusedQKProgramFactory{};
}

void RotaryEmbeddingLlamaFusedQKDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void RotaryEmbeddingLlamaFusedQKDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& q_input_tensor = tensor_args.q_input;
    const auto& k_input_tensor = tensor_args.k_input;
    const auto& cos = tensor_args.cos;
    const auto& sin = tensor_args.sin;
    const auto& trans_mat = tensor_args.trans_mat;

    auto* ref_device = q_input_tensor.device();

    auto validate_tensor = [ref_device](const Tensor& tensor, const std::string& name) {
        TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{} tensor must be on device!", name);
        TT_FATAL(tensor.buffer() != nullptr, "{} tensor must be allocated in buffers on device!", name);
        TT_FATAL(tensor.device() == ref_device, "{} tensor must be on same device!", name);
        TT_FATAL(
            tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "{} tensor must be HEIGHT_SHARDED.",
            name);
        TT_FATAL(tensor.dtype() == DataType::BFLOAT16, "{} tensor must be bfloat16", name);
    };

    validate_tensor(q_input_tensor, "Q input");
    validate_tensor(k_input_tensor, "K input");
    validate_tensor(cos, "Cos");
    validate_tensor(sin, "Sin");
    validate_tensor(trans_mat, "Trans mat");

    Layout tensor_layout = args.row_major_QK ? Layout::ROW_MAJOR : Layout::TILE;
    auto validate_layout = [tensor_layout](const Tensor& tensor, const std::string& name) {
        TT_FATAL(tensor.layout() == tensor_layout, "{} tensor must be in layout {}", name, tensor_layout);
    };
    validate_layout(q_input_tensor, "Q input");
    validate_layout(k_input_tensor, "K input");
    validate_layout(cos, "cos");
    validate_layout(sin, "sin");

    // Check for decode mode
    TT_FATAL(
        q_input_tensor.logical_shape()[0] == 1 && k_input_tensor.logical_shape()[0] == 1,
        "rotary_embedding_llama_fused_qk currently only supports decode mode with seq_len=1.");

    TT_FATAL(
        q_input_tensor.logical_shape()[-1] == k_input_tensor.logical_shape()[-1],
        "Q input tensor and K input tensor must have same head dimensions");
    uint32_t head_dim = q_input_tensor.logical_shape()[-1];
    TT_FATAL(
        head_dim <= 128 ||
            std::get<ttnn::WormholeComputeKernelConfig>(args.compute_kernel_config).fp32_dest_acc_en == false,
        "If head_dim is > 128, fp32_dest_acc_en must be False");

    if (args.row_major_QK) {
        TT_FATAL(
            q_input_tensor.logical_shape()[-2] * q_input_tensor.logical_shape()[-1] == TILE_WIDTH * TILE_HEIGHT,
            "For row major, Q input tensor must be wrapped to tile size");
        TT_FATAL(
            k_input_tensor.logical_shape()[-2] * k_input_tensor.logical_shape()[-1] == TILE_WIDTH * TILE_HEIGHT,
            "For row major, K input tensor must be wrapped to tile size");
    }

    // Check that head_dim is a multiple of 32
    TT_FATAL(head_dim % TILE_WIDTH == 0, "Head dim must be a multiple of TILE_WIDTH");

    TT_FATAL(
        q_input_tensor.memory_config().memory_layout() == args.q_output_mem_config.memory_layout(),
        "Q Input tensor and Q output tensor must have same memory layout");
    TT_FATAL(
        k_input_tensor.memory_config().memory_layout() == args.k_output_mem_config.memory_layout(),
        "K Input tensor and K output tensor must have same memory layout");

    // check that q and k have same batch size and lesser that equal to 32
    uint32_t q_batch_size = q_input_tensor.logical_shape()[1];
    uint32_t k_batch_size = k_input_tensor.logical_shape()[1];
    TT_FATAL(q_batch_size == k_batch_size, "Q and K must have the equal batch size");
    TT_FATAL(
        q_batch_size <= 32,
        "Q and K must have batch size less than or equal to 32, due to parallelization over core-grid of 64");
    uint32_t q_num_cores = q_input_tensor.shard_spec()->grid.num_cores();
    uint32_t k_num_cores = k_input_tensor.shard_spec()->grid.num_cores();
    TT_FATAL(q_num_cores + k_num_cores <= 64, "Q and K must not exceed max core grid size of 64");

    bool is_overlap = q_input_tensor.shard_spec()->grid.intersects(k_input_tensor.shard_spec()->grid);
    TT_FATAL(!is_overlap, "Q and K must not overlap");

    // Check that cos and sin have same dims
    TT_FATAL(cos.logical_shape() == sin.logical_shape(), "Cos and Sin dims must match");
    uint32_t cos_sin_batch_size = cos.logical_shape()[1];
    TT_FATAL(
        cos_sin_batch_size == (q_batch_size + k_batch_size),
        "Cos and Sin are repeated for Q and K, so they must have the same batch size as the sum of Q and K batch "
        "sizes");

    // Checks for transformation matrix
    uint32_t trans_mat_num_cores = trans_mat.shard_spec()->grid.num_cores();
    TT_FATAL((trans_mat.layout() == Layout::TILE), "transformation matrix must be tilized");
    TT_FATAL(
        trans_mat_num_cores >= (q_num_cores + k_num_cores),
        "Transformation matrix is repeated for Q and K must be sharded over core grid of Q and K");
    TT_FATAL(
        trans_mat.shard_spec()->shape[0] == TILE_HEIGHT && trans_mat.shard_spec()->shape[1] == TILE_WIDTH,
        "Transformation matrix must be sharded to single tile of shape (32, 32)");
}

spec_return_value_t RotaryEmbeddingLlamaFusedQKDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& q_input_tensor = tensor_args.q_input;
    const auto& k_input_tensor = tensor_args.k_input;
    const auto& q_shape = q_input_tensor.logical_shape();
    const auto& k_shape = k_input_tensor.logical_shape();
    return {
        TensorSpec(
            q_shape,
            TensorLayout(q_input_tensor.dtype(), PageConfig(q_input_tensor.layout()), args.q_output_mem_config)),
        TensorSpec(
            k_shape,
            TensorLayout(k_input_tensor.dtype(), PageConfig(k_input_tensor.layout()), args.k_output_mem_config))};
}

tensor_return_value_t RotaryEmbeddingLlamaFusedQKDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto [spec_q, spec_k] = compute_output_specs(operation_attributes, tensor_args);
    return {
        create_device_tensor(spec_q, tensor_args.q_input.device()),
        create_device_tensor(spec_k, tensor_args.k_input.device())};
}

}  // namespace ttnn::operations::experimental::transformer::rotary_embedding_llama_fused_qk

namespace ttnn::prim {

ttnn::operations::experimental::transformer::rotary_embedding_llama_fused_qk::tensor_return_value_t
rotary_embedding_llama_fused_qk(
    const Tensor& q_input_tensor,
    const Tensor& k_input_tensor,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const Tensor& trans_mat,
    const tt::tt_metal::MemoryConfig& q_output_mem_config,
    const tt::tt_metal::MemoryConfig& k_output_mem_config,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    bool row_major_QK) {
    using OperationType = ttnn::operations::experimental::transformer::rotary_embedding_llama_fused_qk::
        RotaryEmbeddingLlamaFusedQKDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .q_output_mem_config = q_output_mem_config,
        .k_output_mem_config = k_output_mem_config,
        .compute_kernel_config = compute_kernel_config,
        .row_major_QK = row_major_QK,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .q_input = q_input_tensor,
        .k_input = k_input_tensor,
        .cos = cos_cache,
        .sin = sin_cache,
        .trans_mat = trans_mat,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
