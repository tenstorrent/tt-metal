// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/udm/block_circular_buffer.hpp"
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::udm {

tt::tt_metal::CBHandle CreateBlockCircularBuffer(
    BlockProgram& program, const std::vector<Gcore>& gcores, const tt::tt_metal::CircularBufferConfig& config) {
    // TODO: Implement circular buffer creation across gcores
    TT_FATAL(false, "CreateBlockCircularBuffer not yet implemented");
    return tt::tt_metal::CBHandle();
}

}  // namespace tt::tt_metal::udm
