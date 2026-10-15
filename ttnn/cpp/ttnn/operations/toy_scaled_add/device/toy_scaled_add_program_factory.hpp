// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/distributed/types.hpp"
#include "toy_scaled_add_device_operation_types.hpp"

namespace ttnn::operations::toy_scaled_add {

// Both factories push their kernels in this order, so a kernel's index in the cached Program is its
// value here.
namespace kernel_index {
enum : uint32_t { READER, WRITER, COMPUTE };
}  // namespace kernel_index

// Each factory has two halves:
//   create_descriptor          runs on a program-cache miss and plans everything: the work split,
//                              the circular buffers, every kernel argument;
//   override_runtime_arguments runs on every cache hit instead, and writes only what can differ
//                              between calls that share the program — the buffer addresses and
//                              alpha. Everything else is fixed by the cache key, so a hit leaves
//                              it as the miss wrote it.

// a, b and the output interleaved: tile-rows split over the worker grid, streamed through
// TensorAccessors.
struct InterleavedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ToyScaledAddParams& attrs, const ToyScaledAddInputs& t, Tensor& output);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const ToyScaledAddParams& attrs,
        const ToyScaledAddInputs& t,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

// a, b and the output height-sharded on one shard spec: each core's circular buffers are backed by
// its shards.
struct HeightShardedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ToyScaledAddParams& attrs, const ToyScaledAddInputs& t, Tensor& output);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const ToyScaledAddParams& attrs,
        const ToyScaledAddInputs& t,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::operations::toy_scaled_add
