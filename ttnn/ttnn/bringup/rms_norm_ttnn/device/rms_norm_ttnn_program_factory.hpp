// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "rms_norm_ttnn_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::bringup::rms_norm_ttnn {

// The host program builder: a line-for-line port of rms_norm_ttnn_program_descriptor.py's
// create_program_descriptor().  Same kernels, CT/RT args, defines, CBs, semaphores, cores and
// work split, for every supported cell.  Exposed on its own so the parity test can build the
// program both ways without dispatching.
tt::tt_metal::ProgramDescriptor create_program_descriptor(
    const Tensor& input,
    const Tensor& output,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& residual,
    double epsilon,
    const tt::tt_metal::ComputeConfigDescriptor& compute_config,
    uint32_t subblock_w);

struct RmsNormProgramFactory {
    // Cache miss: build the whole descriptor.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RmsNormParams& operation_attributes, const RmsNormInputs& tensor_args, Tensor& output);

    // Cache hit: only the buffer addresses can change (every other input to the builder is in the
    // program hash), so patch the address runtime-arg slots and re-point the shard-backed CBs.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const RmsNormParams& operation_attributes,
        const RmsNormInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::operations::bringup::rms_norm_ttnn
