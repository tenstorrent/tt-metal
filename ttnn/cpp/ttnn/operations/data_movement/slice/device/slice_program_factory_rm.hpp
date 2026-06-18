// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"

namespace ttnn::operations::data_movement {

// Per-core runtime args for the row-major slice path, derived purely from (args, input, output).
// Single source of truth for both SliceRmProgramFactory::create_descriptor() (cache miss) and
// SliceDeviceOperation::get_dynamic_runtime_args() (cache-hit re-apply). The reader/writer buffer
// address slots are stored at the recorded *_addr_index; create_descriptor() bakes the live address
// there and get_dynamic_runtime_args() re-applies it on every dispatch (the work-core set can grow on
// a cache hit, so every core's address slot must be refreshed). All other slots are hash-folded via
// padded_shape and are not re-applied.
struct SliceRmPerCoreArgs {
    CoreRangeSet all_cores;        // the full set the descriptor binds (kernel / CB core_ranges)
    std::vector<CoreCoord> cores;  // corerange_to_cores(all_cores); every core carries an address slot
    std::vector<std::vector<uint32_t>> reader_args;
    std::vector<std::vector<uint32_t>> writer_args;
    uint32_t reader_addr_index = 0;  // reader arg index holding start_addr + begins_bytes - misalignment
    uint32_t writer_addr_index = 0;  // writer arg index holding the output buffer address

    // Work-split tile counts needed by create_descriptor() for CB sizing (hash-folded, not re-applied).
    uint32_t num_sticks_per_core_group_1 = 0;
    uint32_t num_sticks_per_core_group_2 = 0;
};

SliceRmPerCoreArgs compute_slice_rm_per_core_args(
    const ttnn::prim::SliceParams& args, const Tensor& input, Tensor& output);

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {

struct SliceRmProgramFactory {
    // Contract (1): per-coord ProgramDescriptor.  The src0 CB's total_size /
    // page_size depend on slice_start (via misalignment / unpadded_row_size_bytes),
    // so padded_shape is folded into compute_program_hash() — each unique CB
    // sizing keeps its own cache entry.  On cache hit the framework copies
    // runtime args and patches dynamic CB addresses; CB total_size/page_size
    // are not re-applied (the cached descriptor already carries them).
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim
