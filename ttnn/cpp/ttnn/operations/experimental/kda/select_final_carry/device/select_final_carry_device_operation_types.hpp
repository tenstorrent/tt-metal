// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// A projection run in the final carry's dispatch on the worker cores the carry leaves (nothing reads the carry, so
// its fabric wait hides behind the projection's compute): projection_input @ projection_weight, ttnn.linear's 2D
// multicast matmul with program_config, on its cores.
struct FinalCarryProjectionParams {
    ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig program_config;
    DeviceComputeKernelConfig compute_kernel_config;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct SelectFinalCarryParams {
    uint32_t sequence_parallel_axis;
    uint32_t local_rows;
    uint32_t num_links;
    tt::tt_fabric::Topology topology;
    tt::tt_metal::MemoryConfig output_mem_config;
    // The prefix's final state is valid only on the last chronological rank (the SP relay leaves it there), so
    // without a split tail that rank multicasts it instead of every rank copying its own.
    bool prefix_on_last_rank = false;
    // The worker cores, when given: the first num_links send over the fabric and all of them copy.
    std::optional<std::vector<tt::tt_metal::CoreCoord>> cores;
    // L1 addresses of the caller's barrier and arrival global semaphores; without them each cached workload allocates
    // its own.
    std::optional<std::array<uint32_t, 2>> semaphore_addresses;
    std::optional<FinalCarryProjectionParams> projection;
};

struct SelectFinalCarryInputs {
    Tensor rank_final;
    Tensor prefix_final;
    Tensor actual_start;
    std::optional<Tensor> actual_end;
    std::optional<Tensor> projection_input;
    std::optional<Tensor> projection_weight;
};

}  // namespace ttnn::experimental::prim
