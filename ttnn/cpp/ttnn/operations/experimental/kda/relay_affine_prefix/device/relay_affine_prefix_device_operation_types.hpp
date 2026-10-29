// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// A projection gate_input @ gate_weight run in the relay's dispatch on cores it leaves idle (ttnn.linear's 2D multicast
// matmul with program_config, on its cores): the output's first output_column tile columns are copied from
// gate_copy_source (from copy_source_column) by copy_cores.
struct AffineRelayOutputGateParams {
    ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig program_config;
    DeviceComputeKernelConfig compute_kernel_config;
    tt::tt_metal::MemoryConfig output_mem_config;
    uint32_t output_column = 0;
    uint32_t copy_source_column = 0;
    std::vector<tt::tt_metal::CoreCoord> copy_cores;
};

struct RelayAffinePrefixParams {
    uint32_t sequence_parallel_axis;
    uint32_t local_rows;
    uint32_t num_links;
    // One core per head, then one sender core per link.
    std::vector<tt::tt_metal::CoreCoord> cores;
    tt::tt_metal::MemoryConfig entry_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
    // L1 addresses of the caller's entry-state and final-state arrival counters, kept alive across calls: a peer
    // counts into them before this rank's call starts, and each call subtracts the columns it consumed.
    std::array<uint32_t, 2> semaphore_addresses{};
    std::optional<AffineRelayOutputGateParams> output_gate;
};

struct RelayAffinePrefixInputs {
    Tensor packed;
    Tensor initial_state;
    Tensor actual_start;
    // L1, height-sharded one K x K state per head core and kept alive across calls: the predecessor writes a head's
    // entry state into entry_landing, this rank's next state is staged in state_staging, and the last rank's final
    // state lands in the first rank's final_landing. Consecutive calls need a barrier across the SP line between
    // them: a rank's writer copies entry_landing and final_landing out at the end of its call, and its predecessor's
    // next call overwrites them as soon as it starts.
    Tensor entry_landing;
    Tensor state_staging;
    Tensor final_landing;
    std::optional<Tensor> gate_input;
    std::optional<Tensor> gate_weight;
    std::optional<Tensor> gate_copy_source;
};

}  // namespace ttnn::experimental::prim
