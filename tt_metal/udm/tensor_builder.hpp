// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "tt_metal/udm/types.hpp"
#include "tt_metal/udm/block_builder.hpp"
#include "tt_metal/udm/block_tensor_accessor.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::udm {

/**
 * @brief Top-level builder for UDM programs from distributed tensors
 *
 * TensorBuilder is the main entry point for creating UDM programs.
 * It takes a distributed tensor and creates the necessary block and grid abstractions.
 */
class TensorBuilder {
public:
    /**
     * @brief Create a TensorBuilder from a distributed tensor
     *
     * @param input_tensor The distributed tensor to build from
     * @param grid Grid per grid (empty dimensions = use grid default)
     * @param block Block shape (empty dimensions = use actual mesh shape)
     */
    explicit TensorBuilder(const ttnn::Tensor& input_tensor, const Grid& grid = Grid{}, const Block& block = Block{});

    ~TensorBuilder();

    // Delete copy, allow move
    TensorBuilder(const TensorBuilder&) = delete;
    TensorBuilder& operator=(const TensorBuilder&) = delete;
    TensorBuilder(TensorBuilder&&) noexcept;
    TensorBuilder& operator=(TensorBuilder&&) noexcept;

    /**
     * @brief Get the BlockBuilder
     */
    const BlockBuilder& block_builder() const;
    BlockBuilder& block_builder();

    /**
     * @brief Get the BlockTensorAccessor
     */
    const BlockTensorAccessor& tensor_accessor() const;
    BlockTensorAccessor& tensor_accessor();

    /**
     * @brief Get the input tensor
     */
    const ttnn::Tensor& tensor() const;

    /**
     * @brief Get the reconstructed global tensor shape
     */
    const tt::tt_metal::Shape& global_shape() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Create a TensorBuilder from a distributed tensor
 *
 * @param input_tensor The distributed tensor
 * @param grid Grid per grid (empty dimensions = use grid default)
 * @param block Block shape (empty dimensions = use actual mesh shape)
 *
 * Example:
 *   CreateTensorBuilder(tensor)  // Use defaults
 *   CreateTensorBuilder(tensor, Grid{{1, 16}})  // Custom grid, default block
 *   CreateTensorBuilder(tensor, Grid{{1, 16}}, Block{{1, 4}})  // Custom grid and block
 */
TensorBuilder CreateTensorBuilder(
    const ttnn::Tensor& input_tensor, const Grid& grid = Grid{}, const Block& block = Block{});

}  // namespace tt::tt_metal::udm
