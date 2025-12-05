// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "tt_metal/udm/types.hpp"
#include "tt_metal/udm/block_builder.hpp"
#include "tt_metal/api/tt-metalium/program.hpp"

namespace tt::tt_metal::udm {

/**
 * @brief Multi-device program abstraction
 *
 * BlockProgram represents a program that operates across multiple devices
 * in a virtualized Block configuration.
 */
class BlockProgram {
public:
    /**
     * @brief Create a BlockProgram from a TensorBuilder
     *
     * @param builder The TensorBuilder containing device and tensor information
     */
    explicit BlockProgram(const class TensorBuilder& builder);

    ~BlockProgram();

    // Delete copy, allow move
    BlockProgram(const BlockProgram&) = delete;
    BlockProgram& operator=(const BlockProgram&) = delete;
    BlockProgram(BlockProgram&&) noexcept;
    BlockProgram& operator=(BlockProgram&&) noexcept;

    /**
     * @brief Get the underlying single-device program
     *
     * For now, we may use a single Program object that gets replicated
     * across devices, or maintain multiple Program objects internally.
     */
    tt::tt_metal::Program& program();
    const tt::tt_metal::Program& program() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Create a BlockProgram from a TensorBuilder
 */
BlockProgram CreateBlockProgram(const class TensorBuilder& builder);

}  // namespace tt::tt_metal::udm
