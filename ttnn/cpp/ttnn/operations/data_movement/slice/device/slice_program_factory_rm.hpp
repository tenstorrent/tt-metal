// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

#include <tt-metalium/host_api.hpp>
#include <optional>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"

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

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const SliceParams& args,
        const SliceInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::prim
