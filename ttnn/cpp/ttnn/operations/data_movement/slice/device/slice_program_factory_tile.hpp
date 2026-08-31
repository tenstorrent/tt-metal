// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include <tt-metalium/host_api.hpp>
#include <optional>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"

namespace ttnn::prim {

struct SliceTileProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const SliceParams& args,
        const SliceInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

// Per-core scalars are hash-excluded; a divergent-partition cache hit leaves them stale -> all-zero output (#52651).
std::vector<tt::tt_metal::DynamicRuntimeArg> slice_tile_dynamic_args(
    const SliceParams& args,
    const SliceInputs& tensor_args,
    const Tensor& output,
    uint32_t start_offset,
    uint32_t reader_kernel_idx,
    uint32_t writer_kernel_idx);

}  // namespace ttnn::prim
