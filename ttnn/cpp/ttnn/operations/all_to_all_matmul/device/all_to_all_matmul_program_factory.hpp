// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/distributed/types.hpp"
#include "all_to_all_matmul_device_operation_types.hpp"

namespace ttnn::operations::all_to_all_matmul {

// NoC choices of the source (compute reader NoC1, compute writer NoC0 for DRAM and L1 output, port reader NoC0,
// port sender NoC1). The source's NOC0 is NOC::RISCV_1_default and NOC1 is NOC::RISCV_0_default.
constexpr tt::tt_metal::NOC kComputeReaderNoc = tt::tt_metal::NOC::RISCV_0_default;
constexpr tt::tt_metal::NOC kComputeWriterNoc = tt::tt_metal::NOC::RISCV_1_default;
constexpr tt::tt_metal::NOC kPortReaderNoc = tt::tt_metal::NOC::RISCV_1_default;
constexpr tt::tt_metal::NOC kPortSenderNoc = tt::tt_metal::NOC::RISCV_0_default;

// One program per mesh coordinate (the chip's routes and ports differ).
//   create_descriptor          cache miss: looks up mesh_topology / solve_plan / mesh_resources (const&) and
//                              builds the chip's program; every per-core arg comes from the plan.
//   override_runtime_arguments cache hit: writes the common args only (input, weight, output addresses, the
//                              landing scratch base) and re-points cb::A_LANDED to the scratch buffer when it
//                              moved. The key pins everything else.
struct AllToAllMatmulProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const AllToAllMatmulParams& attrs,
        const AllToAllMatmulInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const AllToAllMatmulParams& attrs,
        const AllToAllMatmulInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::operations::all_to_all_matmul
