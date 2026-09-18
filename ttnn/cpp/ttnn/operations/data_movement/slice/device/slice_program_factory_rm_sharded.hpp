// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"

namespace ttnn::prim {

struct SliceRmShardedProgramFactory {
    // Contract (1): per-coord ProgramDescriptor.  Both CBs are sharded
    // (CBDescriptor::buffer bound to input/output buffers); the framework
    // patches the dynamic CB addresses on cache hit via
    // apply_descriptor_runtime_args.  CB total_size/page_size are NOT patched
    // — padded_shape is folded into compute_program_hash() so each unique
    // sizing gets its own cache entry.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);

    // Cache-hit hook: the reader args depend only on shapes/slice_start/shard specs, all cache-keyed,
    // so the only per-dispatch state is the two CB addresses. Patched in O(1); no descriptor rebuild.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const SliceParams& args,
        const SliceInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::prim
