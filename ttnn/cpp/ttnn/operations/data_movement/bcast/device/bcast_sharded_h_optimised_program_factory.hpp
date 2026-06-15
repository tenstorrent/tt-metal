// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bcast_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include <vector>

namespace ttnn::prim {

struct BcastShardedHOptimisedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const BcastParams& operation_attributes, const BcastInputs& tensor_args, Tensor& tensor_return_value);

    // Re-applies the COMPLETE per-core runtime-arg state on a program-cache hit. The reader/compute
    // geometry args (Ht, Wt, offsets, batch_b, ...) are derived from the shard spec / (padded) SHAPE,
    // which the default program hash does NOT fully cover (TensorSpec hashes logical_shape, not
    // padded_shape). Single source of truth shared with create_descriptor().
    static std::vector<tt::tt_metal::DynamicRuntimeArg> get_dynamic_runtime_args(
        const BcastParams& operation_attributes, const BcastInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
