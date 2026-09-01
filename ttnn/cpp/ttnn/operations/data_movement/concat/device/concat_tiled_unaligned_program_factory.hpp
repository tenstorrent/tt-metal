// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "concat_device_operation_types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

// Native width-concat for TILE-layout interleaved tensors whose last dim is not tile-aligned.
//
// Each core owns a range of 32-row bands (tile-rows). Per band it: reads every input's tile-row,
// untilizes it, byte-assembles the logical rows side by side (dropping each input's width padding
// and zero-filling the output's), retilizes, and writes the output tile-row. Peak memory is
// inputs + output + a few KB of CBs per core -- no full-size intermediates and no DRAM staging,
// unlike the untilize/transpose/retilize massaging pipeline this bypasses.
struct ConcatTiledUnalignedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value);
};

// True when ConcatTiledUnalignedProgramFactory can run this concat: last-dim concat of TILE
// interleaved tensors with tile padding on the concat dim, interleaved output, groups == 1,
// supported dtype, and per-core CBs that fit in the L1 window the allocator reports as free
// right now (accounting for every live L1 buffer). Set output_already_allocated to false at
// routing time (before the output buffer exists) so its worst-case per-bank footprint is
// reserved on top, and to true inside the device op, where the launch infra has already
// allocated the output and the free window accounts for it.
bool can_use_tiled_unaligned_concat(
    const std::vector<Tensor>& input_tensors,
    uint32_t normalized_dim,
    unsigned int groups,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    bool output_already_allocated);

}  // namespace ttnn::prim
