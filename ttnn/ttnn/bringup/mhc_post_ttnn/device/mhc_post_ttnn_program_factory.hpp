// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "mhc_post_ttnn_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::bringup::mhc_post_ttnn {

// The host program builder: a line-for-line port of mhc_post_program_descriptor.py's create_program_descriptor().
// Same kernels, CT / RT args, CBs, semaphores, cores and work split. Exposed on its own so the parity test can build
// the program both ways without dispatching.
tt::tt_metal::ProgramDescriptor create_program_descriptor(
    const Tensor& input,
    const Tensor& residual,
    const Tensor& post,
    const Tensor& comb,
    const Tensor& output,
    const tt::tt_metal::ComputeConfigDescriptor& compute_config);

struct MhcPostProgramFactory {
    // Cache miss: build the whole descriptor.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const MhcPostParams& operation_attributes, const MhcPostInputs& tensor_args, Tensor& output);

    // Cache hit: only buffer addresses can change (every other input to the builder is in the program hash), so
    // patch the five address slots of the two data-movement kernels.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const MhcPostParams& operation_attributes,
        const MhcPostInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::operations::bringup::mhc_post_ttnn
