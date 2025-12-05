// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include "tt_metal/udm/types.hpp"
#include "tt_metal/udm/block_program.hpp"
#include "tt_metal/api/tt-metalium/circular_buffer_types.hpp"

namespace tt::tt_metal::udm {

/**
 * @brief Create a circular buffer across multiple global cores
 *
 * @param program The BlockProgram to add the circular buffer to
 * @param gcores The global cores to create the circular buffer on
 * @param config Circular buffer configuration
 * @return CBHandle Handle to the created circular buffer
 */
tt::tt_metal::CBHandle CreateBlockCircularBuffer(
    BlockProgram& program, const std::vector<Gcore>& gcores, const tt::tt_metal::CircularBufferConfig& config);

}  // namespace tt::tt_metal::udm
