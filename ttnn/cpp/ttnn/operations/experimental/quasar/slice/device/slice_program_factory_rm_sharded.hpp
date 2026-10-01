// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/quasar/slice/device/slice_device_operation_types.hpp"

namespace ttnn::prim::qsr {

struct SliceRmShardedProgramFactory {
    // Metal 2.0 port. Both shards are resident borrowed-memory DFBs (dfb::cb_in on the input shard,
    // dfb::cb_out on the output shard), self-looped on the single reader kernel to satisfy the DFB
    // producer-and-consumer requirement. The framework refreshes the borrowed L1 addresses from the
    // bound tensor args on cache hit.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim::qsr
