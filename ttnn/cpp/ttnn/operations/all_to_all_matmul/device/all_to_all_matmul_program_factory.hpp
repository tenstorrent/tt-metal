// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/workload/legacy/kernel_types.hpp>
#include <tt-metalium/workload/program.hpp>
#include <tt-metalium/workload/legacy/program_descriptors.hpp>
#include <tt-metalium/workload/legacy/workload_descriptor.hpp>

#include "ttnn/distributed/types.hpp"
#include "all_to_all_matmul_device_operation_types.hpp"

namespace ttnn::operations::all_to_all_matmul {

// NoC choices of the source (compute reader NoC1, compute writer NoC0 for DRAM and L1 output, port reader NoC0,
// port sender NoC1). The source's NOC0 is NOC::RISCV_1_default and NOC1 is NOC::RISCV_0_default.
constexpr tt::tt_metal::NOC kComputeReaderNoc = tt::tt_metal::NOC::RISCV_0_default;
constexpr tt::tt_metal::NOC kComputeWriterNoc = tt::tt_metal::NOC::RISCV_1_default;
constexpr tt::tt_metal::NOC kPortReaderNoc = tt::tt_metal::NOC::RISCV_1_default;
constexpr tt::tt_metal::NOC kPortSenderNoc = tt::tt_metal::NOC::RISCV_0_default;

// One program per mesh coordinate (the chip's ring position, routes and ports differ), merged by the framework into
// one program for the whole mesh with per-device runtime args (ttnn/per_device_program_merge.hpp): the chips'
// programs differ only in runtime-arg values and in which port cores run the port kernels, so a call dispatches one
// program instead of one per chip. Every per-core arg comes from the plan; the key pins everything but the tensor
// addresses, which the buffer bindings (input, weight, output, the landing scratch and its circular buffer) patch on
// a cache hit.
struct AllToAllMatmulProgramFactory {
    static constexpr bool merge_per_device_programs = true;
    // The kernels read their slot, entry, destination and port counts from their args.
    static constexpr bool pad_per_device_runtime_args = true;

    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllToAllMatmulParams& attrs,
        const AllToAllMatmulInputs& tensor_args,
        Tensor& output,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    // A port core that runs no port on this chip (another chip's does): no entries to read, no connection to open.
    static std::optional<std::vector<uint32_t>> idle_runtime_args(size_t kernel_index, const CoreCoord& core);
};

}  // namespace ttnn::operations::all_to_all_matmul
