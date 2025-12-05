// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/udm/block_kernel.hpp"
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::udm {

BlockKernelHandle CreateBlockKernel(
    BlockProgram& program,
    const std::string& file_name,
    const std::vector<Gcore>& gcores,
    const std::variant<tt::tt_metal::DataMovementConfig, tt::tt_metal::ComputeConfig, tt::tt_metal::EthernetConfig>&
        config) {
    // TODO: Implement kernel creation across gcores
    TT_FATAL(false, "CreateBlockKernel not yet implemented");
    return 0;
}

void SetBlockKernelRuntimeArgs(
    BlockProgram& program, BlockKernelHandle kernel_id, const Gcore& gcore, const std::vector<uint32_t>& runtime_args) {
    // TODO: Implement runtime args setting
    TT_FATAL(false, "SetRuntimeArgs for block kernel not yet implemented");
}

}  // namespace tt::tt_metal::udm
