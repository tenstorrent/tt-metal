// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/distributed/types.hpp"
#include "grouped_matmul_device_operation_types.hpp"

namespace ttnn::operations::grouped_matmul {

// One program for the whole mesh (every chip runs the same replicated call). create_descriptor builds the
// interleaved path (grouped_matmul.py program()) or, with attrs.bank_layout, the bank path (bank_program()).
// Kernels in groups of three (kernel_role in kernels/grouped_matmul_args.hpp). override_runtime_arguments
// rewrites only the common address slots (reader X_ADDR / W_ADDR, writer Y_ADDR) of every group.
struct GroupedMatmulProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const GroupedMatmulParams& attrs, const GroupedMatmulInputs& tensors, Tensor& output);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const GroupedMatmulParams& attrs,
        const GroupedMatmulInputs& tensors,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::operations::grouped_matmul
