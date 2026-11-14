// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "concat_device_operation_types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

// Most inputs one ConcatTiledUnalignedProgramFactory program carries. Above this count the circular
// buffers that hold a whole band of every input no longer fit reliably in one core's L1 window, so
// ttnn::concat groups a longer list across several programs of this factory instead. This is not a
// hardware limit: the enforced bounds are the per-core runtime-arg ceiling (4096 words per Tensix
// kernel, tt_metal/impl/kernels/kernel.cpp) and the free L1 window, both far past this, and both
// already checked at program-build time.
constexpr size_t kMaxTiledUnalignedConcatInputs = 32;

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
// supported dtype, per-core CBs that fit in the L1 window the allocator reports as free
// right now (accounting for every live L1 buffer), and no more inputs than
// kMaxTiledUnalignedConcatInputs. Set output_already_allocated to false at routing time (before the
// output buffer exists) so its worst-case per-bank footprint is reserved on top, and to true inside
// the device op, where the launch infra has already allocated the output and the free window accounts
// for it.
bool can_use_tiled_unaligned_concat(
    const std::vector<Tensor>& input_tensors,
    uint32_t normalized_dim,
    unsigned int groups,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    bool output_already_allocated);

// True when this concat is on the unaligned path apart from how many inputs it carries. A longer list
// than kMaxTiledUnalignedConcatInputs is still served by that factory, one program per group of at
// most kMaxTiledUnalignedConcatInputs inputs, so ttnn::concat can keep it on the on-device path
// instead of falling back to the untilize/transpose/chunk pipeline.
bool is_tiled_unaligned_concat_candidate(
    const std::vector<Tensor>& input_tensors,
    uint32_t normalized_dim,
    unsigned int groups,
    const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::prim
