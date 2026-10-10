// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "../untilize_device_operation_types.hpp"

namespace ttnn::prim::qsr {

// Untilize's multi-thread data movement runs 4 reader threads, which take the sub-blocks compute
// waits for in turn. The sub-block is the widest divisor of the block that fits in half-sync DEST
// and still leaves every reader thread at least one sub-block on the core with the fewest blocks
// (the DFB's final-credit barrier waits for all of a kernel's threads). 0 if no width does.
constexpr uint32_t kUntilizeNumReaderThreads = 4;
inline uint32_t untilize_split_sub_block_tiles(
    uint32_t block_tiles, uint32_t min_blocks_per_core, bool fp32_dest_acc_en) {
    const uint32_t dest_limit = fp32_dest_acc_en ? 4 : 8;
    for (uint32_t w = std::min(dest_limit, block_tiles); w > 0; --w) {
        if (block_tiles % w == 0 && min_blocks_per_core * (block_tiles / w) >= kUntilizeNumReaderThreads) {
            return w;
        }
    }
    return 0;
}

// Untilizes interleaved tile input into a row-major interleaved output of the same width with 4
// reader threads and 2 single-thread writer kernels, all with implicit sync. Any height padding
// must sit in the last tile row; its padding rows are dropped. Returns std::nullopt when the
// tensors do not fit this path.
std::optional<ttnn::device_operation::ProgramArtifacts> create_untilize_split_rows_program(
    const Tensor& input, const Tensor& output, bool fp32_dest_acc_en);

struct UntilizeMultiCoreProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const UntilizeOperationAttributes& operation_attributes,
        const UntilizeTensorArgs& tensor_args,
        UntilizeTensorReturnValue& output);
};
}  // namespace ttnn::prim::qsr
