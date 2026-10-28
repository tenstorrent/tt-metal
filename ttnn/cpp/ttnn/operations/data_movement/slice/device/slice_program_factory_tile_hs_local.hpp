// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"

namespace ttnn::prim {

// TILE, HEIGHT_SHARDED input and output on the same grid with the same shard height, slicing only the last
// dimension at tile granularity: every core copies the selected tile columns of each of its own tile rows with
// local NOC reads (one read per tile row), nothing crosses cores and no page lookups are needed. Both CBs are
// bound to the input and output buffers; their addresses are patched on cache hit.
struct SliceTileHsLocalProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
    static bool applies(const SliceParams& args, const Tensor& input);
};

}  // namespace ttnn::prim
