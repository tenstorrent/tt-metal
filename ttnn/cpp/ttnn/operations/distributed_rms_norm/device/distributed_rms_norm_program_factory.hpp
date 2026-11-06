// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "distributed_rms_norm_device_operation_types.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::distributed_rms_norm {

// One program per mesh coordinate (the chip's exchange schedule differs by position). Kernels are pushed in
// kernel_index order (kernels/distributed_rms_norm_args.hpp): READER, WRITER, COMPUTE, PORT.
//
//   create_descriptor          plans on a miss (detail::solve_plan by const&), builds CBs, semaphores, kernels
//                              and the fabric connection args;
//   override_runtime_arguments on a hit writes the common address slots (input, output, gamma, landing), the
//                              EPS_BITS scalar and re-points the shard-backed CBs (cb::X_TILES, cb::OUTPUT).
struct DistributedRmsNormProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const DistributedRmsNormParams& attrs,
        const DistributedRmsNormInputs& t,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const DistributedRmsNormParams& attrs,
        const DistributedRmsNormInputs& t,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::operations::distributed_rms_norm
