// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/distributed/types.hpp"
#include "groupnorm_sc_N_1_HW_C_b_device_operation_types.hpp"
#include "groupnorm_sc_N_1_HW_C_b_planner.hpp"

namespace ttnn::operations::groupnorm_sc_N_1_HW_C_b {

// The descriptor holds its kernels in the Python order: readers (NoC pairing 0, then 1), writers ((root, 0),
// (root, 1), (member, 0), (member, 1)), computes (root, member); only non-empty groups. A kernel's index in the
// cached Program follows from these counts, which the plan (in the key) fixes.
struct KernelLayout {
    uint32_t num_readers = 0;
    uint32_t num_writers = 0;
    uint32_t num_computes = 0;
    uint32_t first_writer() const { return num_readers; }
    uint32_t first_compute() const { return num_readers + num_writers; }
};
KernelLayout kernel_layout(const planner::Plan& plan);

struct GroupnormScN1HwCbProgramFactory {
    // Cache miss: looks up the memoized plan, builds every CB, kernel and semaphore.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const GroupnormScN1HwCbParams& attrs, const GroupnormScN1HwCbInputs& tensors, Tensor& output);

    // Cache hit: rewrites the per-core address slots (reader IN/GAMMA/BETA/OUT, writer OUT), compute EPS_BITS (left
    // out of the key), and re-points the shard-backed CBs by CB index, on every hit.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const GroupnormScN1HwCbParams& attrs,
        const GroupnormScN1HwCbInputs& tensors,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::operations::groupnorm_sc_N_1_HW_C_b
