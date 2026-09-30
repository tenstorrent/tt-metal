// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "mhc_pre_ttnn_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::bringup::mhc_pre_ttnn {

// The host program builder: a line-for-line port of mhc_pre_program_descriptor.py's make_plan() and
// create_program_descriptor(). Same group geometry and block solve, kernels (one reader / writer pair per NoC
// placement set), CT / RT args, CBs (incl. aliases), multicast wires, semaphores and defines. Exposed on its own so
// the parity test can build the program both ways without dispatching.
tt::tt_metal::ProgramDescriptor create_program_descriptor(
    const Tensor& x,
    const Tensor& w,
    const Tensor& b,
    const Tensor& y,
    const Tensor& post,
    const Tensor& comb,
    const MhcPreParams& params);

struct MhcPreProgramFactory {
    using Outputs = std::tuple<Tensor, Tensor, Tensor>;

    // Cache miss: build the whole descriptor.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const MhcPreParams& operation_attributes, const MhcPreInputs& tensor_args, Outputs& outputs);

    // Cache hit: only buffer addresses can change (every other input to the builder is in the program hash), so
    // patch the address slots of every reader / writer kernel.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const MhcPreParams& operation_attributes,
        const MhcPreInputs& tensor_args,
        Outputs& outputs,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::operations::bringup::mhc_pre_ttnn
