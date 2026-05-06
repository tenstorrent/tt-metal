// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bit>
#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/matmul/shared_with_host/activation_type.hpp"

namespace ttnn::operations::matmul::utilities {

// Define the buffering depth for input CBs (0 and 1) for mcast variants.
// 2 = double buffer, 3 = triple buffer, etc.
// Allows easily changing buffering strategy in one place for relevant factories.
constexpr uint32_t MCAST_INPUT_BUFFERING_DEPTH = 2;

/**
 * @brief True when fused matmul bias add can use the row-broadcast kernel path.
 *
 * Broadcast applies when there is no distinct row axis (rank < 2, e.g. vector bias) or the
 * logical row dimension is 1 (shape[-2] == 1, e.g. [..., 1, N]). Otherwise the bias has multiple
 * logical rows and the fused kernel must use elementwise add_tiles.
 */
inline bool fused_matmul_bias_row_broadcastable(const std::optional<const Tensor>& bias) {
    if (!bias.has_value()) {
        return false;
    }
    const auto& shape = bias->logical_shape();
    if (shape.rank() < 2) {
        return true;
    }
    return shape[-2] == 1;
}

uint32_t get_estimated_size_of_cbs(
    uint32_t per_core_M,
    uint32_t per_core_N,
    uint32_t in0_block_w,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool transpose_a,
    bool transpose_b,
    uint32_t interm_single_tile_size,
    uint32_t bias_single_tile_size);

uint32_t estimate_interm_tile_size(
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    tt::tt_metal::DataType output_dtype);

uint32_t get_max_l1_space(const tt::tt_metal::Tensor& input_tensor_a);

bool is_input_batched(const ttnn::Shape& shape);

/**
 * @brief Computes the output shape of a matmul operation given two input tensors
 *
 * Determines the output shape based on the broadcasting rules for matrix multiplication:
 * - For 2D tensors: [m, k] @ [k, n] -> [m, n]
 * - For tensors with batch dimensions, the batch dimensions are broadcast
 * - For vector-matrix multiplication (rank 1 @ rank 2), the result is a vector
 *  Takes into account the transpose flags for the input tensors.
 *
 * @param input_tensor_a First input tensor
 * @param input_tensor_b Second input tensor
 * @param transpose_a Whether to transpose the first input tensor
 * @param transpose_b Whether to transpose the second input tensor
 * @return Shape of the resulting tensor after matmul
 */
ttnn::Shape compute_matmul_output_shape(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, bool transpose_a, bool transpose_b);

/*
 * @brief Computes the output shape of a matmul operation with bias given two input shapes
 *
 * Determines the output shape based on the broadcasting rules for matrix multiplication with bias:
 *
 * @param matmul_shape The shape of the matmul operation
 * @param bias_shape The shape of the bias tensor
 * @return Shape of the resulting tensor after matmul with bias
 */
ttnn::Shape compute_matmul_with_bias_output_shape(const ttnn::Shape& matmul_shape, const ttnn::Shape& bias_shape);

using Activation = std::variant<std::string, ttnn::operations::unary::UnaryWithParam>;
std::optional<ttnn::operations::unary::UnaryWithParam> get_fused_activation(
    const std::optional<const Activation>& activation);

tt::tt_metal::Tile get_output_tile(
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const std::optional<const tt::tt_metal::Tile>& output_tile,
    const std::optional<const tt::tt_metal::Tile>& optional_output_tensor_tile);

/**
 * @brief Calculate the M dimension for matmul operations
 *
 * @param padded_shape The padded shape of the tensor
 * @param tile The tile for the tensor (optional)
 * @param fuse_batch Whether to fuse batch dimensions
 * @return uint32_t The calculated M dimension
 */
inline uint32_t get_M_dim(
    const tt::tt_metal::Shape& padded_shape, const std::optional<tt::tt_metal::Tile>& tile, const bool fuse_batch) {
    uint32_t tile_height = tile.has_value() ? tile.value().get_height() : 1;
    if (fuse_batch) {
        return padded_shape.volume() / padded_shape[-1] / tile_height;
    }

    // Batch dims not fused, so take the height dimension
    return padded_shape[-2] / tile_height;
}

/**
 * @brief Calculate the K dimension for matmul operations
 *
 * @param padded_shape The padded shape of the tensor
 * @param tile The tile for the tensor (optional)
 * @return uint32_t The calculated K dimension
 */
inline uint32_t get_K_dim(const tt::tt_metal::Shape& padded_shape, const std::optional<tt::tt_metal::Tile>& tile) {
    return padded_shape[-1] / (tile.has_value() ? tile.value().get_width() : 1);
}

/**
 * @brief Calculate the N dimension for matmul operations
 *
 * @param padded_shape The padded shape of the tensor
 * @param tile The tile for the tensor (optional)
 * @return uint32_t The calculated N dimension
 */
inline uint32_t get_N_dim(const tt::tt_metal::Shape& padded_shape, const std::optional<tt::tt_metal::Tile>& tile) {
    return padded_shape[-1] / (tile.has_value() ? tile.value().get_width() : 1);
}

struct In0TransposeStrides {
    uint32_t stride_w;
    uint32_t stride_h;
};

/**
 * @brief Compute in0 tensor strides for the reader kernel, handling transpose_a with fuse_batch.
 *
 * When transpose_a is true and batches are not fused (M == M_per_batch), the tile traversal
 * order must be transposed: stride_w = M_per_batch, stride_h = 1. When batches are fused
 * (M != M_per_batch) or transpose_a is false, normal strides apply: stride_w = 1, stride_h = K.
 *
 * The caller must ensure that transpose_a + fuse_batch + M_per_batch > 1 is not used; this
 * unsupported combination is validated in MatmulDeviceOperation::validate_on_program_cache_miss.
 *
 * @param M Total M dimension in tiles (possibly fused across batches)
 * @param M_per_batch M dimension in tiles for a single batch
 * @param transpose_a Whether the A tensor is logically transposed
 * @param K K dimension in tiles
 * @return In0TransposeStrides with stride_w and stride_h
 */
inline In0TransposeStrides get_in0_transpose_strides(uint32_t M, uint32_t M_per_batch, bool transpose_a, uint32_t K) {
    const bool batch_fused = (M != M_per_batch);
    return {
        .stride_w = (transpose_a && !batch_fused) ? M_per_batch : 1u,
        .stride_h = (transpose_a && !batch_fused) ? 1u : K,
    };
}

/**
 * @brief Get the padded shape of a tensor, with optional transpose.
 *
 * Returns the padded shape of the tensor. If transpose is true, the padded shape dimensions are swapped.
 *
 * @param input_tensor The tensor whose padded shape is queried.
 * @param transpose Whether to return the padded shape after transposing (swap height and width).
 * @return ttnn::Shape The padded shape of the tensor, possibly transposed.
 */
inline ttnn::Shape get_matmul_tensor_padded_shape(const Tensor& input_tensor, bool transpose) {
    auto padded_shape = input_tensor.padded_shape();
    if (transpose) {
        std::swap(padded_shape[-2], padded_shape[-1]);
    }
    return padded_shape;
}

/**
 * @brief Get the tile shape of a tensor, with optional transpose.
 *
 * Returns a tuple representing the height and width of the tensor's tile. If transpose is true,
 * the tile shape dimensions are swapped.
 *
 * @param input_tensor The tensor whose tile shape is queried.
 * @param transpose Whether to return the tile shape after transposing (swap height and width).
 * @return tt::tt_metal::Tile The tile shape of the tensor, possibly transposed.
 */
tt::tt_metal::Tile get_matmul_tile(const Tensor& input_tensor, bool transpose);

/**
 * @brief Get the shape of a tensor, with optional transpose.
 *
 * Returns the shape of the tensor. If transpose is true, the shape dimensions are swapped.
 *
 * @param input_tensor The tensor whose shape is queried.
 * @param transpose Whether to return the shape after transposing (swap height and width).
 * @return ttnn::Shape The shape of the tensor, possibly transposed.
 */
inline ttnn::Shape get_matmul_tensor_logical_shape(const Tensor& input_tensor, bool transpose) {
    auto shape = input_tensor.logical_shape();
    if (transpose) {
        std::swap(shape[-2], shape[-1]);
    }
    return shape;
}

inline KernelActivation get_activation_type(ttnn::operations::unary::UnaryOpType opType) {
    using ttnn::operations::unary::UnaryOpType;
    switch (opType) {
        case UnaryOpType::GELU: return KernelActivation::GELU;
        case UnaryOpType::TANH: return KernelActivation::TANH;
        case UnaryOpType::SILU: return KernelActivation::SILU;
        case UnaryOpType::RELU6: return KernelActivation::RELU6;
        case UnaryOpType::SIGMOID: return KernelActivation::SIGMOID;
        case UnaryOpType::HARDSIGMOID: return KernelActivation::HARDSIGMOID;
        case UnaryOpType::HARDTANH: return KernelActivation::HARDTANH;
        case UnaryOpType::SELU: return KernelActivation::SELU;
        case UnaryOpType::SOFTPLUS: return KernelActivation::SOFTPLUS;
        default: TT_THROW("Unsupported UnaryOpType for fused activation: {}", opType);
    };
}

/**
 * @brief Consolidated activation parameters structure
 *
 * Contains the activation type and its associated parameters in a single struct
 * These values are passed as compile time arguments to the kernel
 */
struct ActivationParams {
    KernelActivation type = KernelActivation::NONE;
    uint32_t param0 = 0;
    uint32_t param1 = 0;
    uint32_t param2 = 0;
};

/**
 * @brief Extract activation parameters
 *
 * Extracts the activation type and both parameters. Prepares parameter values for the kernel.
 *
 * @param activation The UnaryWithParam containing the activation operation and parameters
 * @return ActivationParams struct with type and activation specific param0, param1, param2
 */
inline ActivationParams get_activation_params(const ttnn::operations::unary::UnaryWithParam& activation) {
    using ttnn::operations::unary::UnaryOpType;

    // Activation parameters provided by the ttnn op.
    std::span<const float> params = activation.get_params();
    TT_FATAL(
        params.size() <= 2, "Invalid number of activation parameters: {}. Expected no more than 2.", params.size());
    const bool has_first = !params.empty();
    const bool has_second = params.size() > 1;

    // Activation parameters to be given to the kernel
    ActivationParams result;

    switch (activation.op_type) {
        case UnaryOpType::GELU:
            result.type = KernelActivation::GELU;
            // param0 is vector mode (0=RC, 1=R, 2=C) or fast mode
            result.param0 = has_first ? static_cast<uint32_t>(params[0]) : 0;
            break;

        case UnaryOpType::TANH:
            result.type = KernelActivation::TANH;
            // param0 is vector mode or fast mode
            result.param0 = has_first ? static_cast<uint32_t>(params[0]) : 0;
            break;

        case UnaryOpType::SILU:
            result.type = KernelActivation::SILU;
            // No parameters currently
            break;

        case UnaryOpType::RELU6:
            result.type = KernelActivation::RELU6;
            // param0 is max value (default 6.0)
            result.param0 = has_first ? std::bit_cast<uint32_t>(params[0]) : 0x40c00000u;
            break;
        case UnaryOpType::SIGMOID: {
            result.type = KernelActivation::SIGMOID;
            const uint32_t param0 = has_first ? static_cast<uint32_t>(params[0]) : 0;
            TT_FATAL(param0 <= 4, "Invalid Vector mode value: {}", param0);
            result.param0 = param0;
            // param1 is fast_approximate flag
            result.param1 = has_second ? static_cast<uint32_t>(params[1]) : 0;
            break;
        }

        case UnaryOpType::HARDSIGMOID:
            result.type = KernelActivation::HARDSIGMOID;
            // param0 could support approximation mode
            result.param0 = has_first ? static_cast<uint32_t>(params[0]) : 0;
            break;

        case UnaryOpType::HARDTANH:
            result.type = KernelActivation::HARDTANH;
            // param0 is min value (default -1.0)
            result.param0 = has_first ? std::bit_cast<uint32_t>(params[0]) : 0xbf800000u;
            // param1 is max value (default 1.0)
            result.param1 = has_second ? std::bit_cast<uint32_t>(params[1]) : 0x3f800000u;
            break;

        case UnaryOpType::SELU:
            result.type = KernelActivation::SELU;
            // param0 is alpha (default 1.67326)
            result.param0 = has_first ? std::bit_cast<uint32_t>(params[0]) : 0x3fd637bdu;
            // param1 is lambda (default 1.05070)
            result.param1 = has_second ? std::bit_cast<uint32_t>(params[1]) : 0x3f8674f5u;
            break;

        case UnaryOpType::SOFTPLUS:
            result.type = KernelActivation::SOFTPLUS;
            // param0 is beta (default 1.0)
            // param1 is threshold (default 20.0)
            // we also prepare beta reciprocal as a kernel compile arg,
            // which is passed as param2
            if (has_first) {
                float beta = params[0];
                TT_FATAL(beta != 0, "SOFTPLUS activation beta parameter cannot be zero");
                float beta_reciprocal = 1.0f / params[0];
                result.param0 = std::bit_cast<uint32_t>(beta);
                result.param2 = std::bit_cast<uint32_t>(beta_reciprocal);
            } else {
                result.param0 = 0x3f800000u;
                result.param2 = 0x3f800000u;
            }
            result.param1 = has_second ? std::bit_cast<uint32_t>(params[1]) : 0x41a00000u;
            break;

        default: TT_THROW("Unsupported UnaryOpType for fused activation: {}", activation.op_type);
    }

    return result;
}

}  // namespace ttnn::operations::matmul::utilities

namespace ttnn::prim::dram_sharded_helpers {
// This type of access pattern cannot be copied.
// Treat it as a one off patch to restore functionality that
// was adjusted to fix one P0 causing another P0.
// TODO: Proper fix will be implemented in Issue #32205
tt::tt_metal::IDevice* get_device_for_dram_banks(const ttnn::Tensor& a, const ttnn::MeshCoordinate& coord);

void get_max_page_size_and_num_pages(
    tt::tt_metal::IDevice* device, uint32_t num_tiles, uint32_t tile_size, uint32_t& page_size, uint32_t& num_pages);

void move_common_entries(std::vector<CoreCoord>& v1, std::vector<CoreCoord>& v2, std::vector<CoreCoord>& commons);

void get_optimal_dram_bank_to_reader_assignment(
    tt::tt_metal::IDevice* device,
    std::vector<CoreCoord>& all_worker_cores_ordered,
    CoreRangeSet& all_worker_cores,
    tt::tt_metal::NOC noc);

}  // namespace ttnn::prim::dram_sharded_helpers
