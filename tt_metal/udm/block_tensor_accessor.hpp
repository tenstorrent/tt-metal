// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>
#include "tt_metal/udm/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::udm {

/**
 * @brief Accessor for distributed tensors in UDM programs
 *
 * BlockTensorAccessor provides a global view of tensor access patterns
 * across multiple grids, handling address translation and compile-time
 * arguments for kernels.
 */
class BlockTensorAccessor {
public:
    /**
     * @brief Create a BlockTensorAccessor from tensor and block builder
     *
     * @param tensor The distributed tensor
     * @param block_builder The BlockBuilder containing grid mappings
     */
    BlockTensorAccessor(const ttnn::Tensor& tensor, const class BlockBuilder& block_builder);

    ~BlockTensorAccessor();

    // Delete copy, allow move
    BlockTensorAccessor(const BlockTensorAccessor&) = delete;
    BlockTensorAccessor& operator=(const BlockTensorAccessor&) = delete;
    BlockTensorAccessor(BlockTensorAccessor&&) noexcept;
    BlockTensorAccessor& operator=(BlockTensorAccessor&&) noexcept;

    /**
     * @brief Get compile-time arguments for tensor access in kernels
     */
    std::vector<uint32_t> get_compile_time_args() const;

    /**
     * @brief Get the buffer address
     */
    uint64_t get_buffer_address() const;

    /**
     * @brief Get the aligned page size
     */
    uint32_t get_aligned_page_size() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::udm
