// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "move_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Address-derived reader args for the in-place sharded move (chunk size = output_addr - input_addr).
// Single source of truth: computed here once and used both by the factory at build time and by
// MoveDeviceOperation::get_dynamic_runtime_args to re-apply them on every cache hit.
struct MoveShardedReaderArgs {
    uint32_t total_size_bytes;
    uint32_t num_chunks;
    uint32_t move_chunk_size_bytes;
    uint32_t remainder_chunk_size_bytes;
};
MoveShardedReaderArgs compute_move_sharded_reader_args(const Tensor& input, const Tensor& output);

struct MoveShardedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const MoveOperationAttributes& operation_attributes,
        const MoveTensorArgs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
