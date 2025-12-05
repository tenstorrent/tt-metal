// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include "tt_metal/udm/types.hpp"
#include "tt_metal/udm/block_program.hpp"
#include "tt_metal/api/tt-metalium/kernel_types.hpp"

namespace tt::tt_metal::udm {

/**
 * @brief Create a block kernel that operates across multiple grids
 *
 * @param program The BlockProgram to add the kernel to
 * @param file_name Path to the kernel source file
 * @param gcores The global cores to place the kernel on
 * @param config Kernel configuration (DataMovementConfig, ComputeConfig, or EthernetConfig)
 * @return BlockKernelHandle Handle to the created kernel
 */
BlockKernelHandle CreateBlockKernel(
    BlockProgram& program,
    const std::string& file_name,
    const std::vector<Gcore>& gcores,
    const std::variant<tt::tt_metal::DataMovementConfig, tt::tt_metal::ComputeConfig, tt::tt_metal::EthernetConfig>&
        config);

/**
 * @brief Set runtime arguments for a specific gcore
 *
 * @param program The BlockProgram containing the kernel
 * @param kernel_id The kernel handle
 * @param gcore The global core to set arguments for
 * @param runtime_args The runtime arguments
 */
void SetBlockKernelRuntimeArgs(
    BlockProgram& program, BlockKernelHandle kernel_id, const Gcore& gcore, const std::vector<uint32_t>& runtime_args);

}  // namespace tt::tt_metal::udm
