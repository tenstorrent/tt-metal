// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/operation.hpp"
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
    // The caller's entry-state and final-state arrival semaphores: a peer can bump them before this call starts.
    std::array<uint32_t, 2> semaphore_addresses{};
    std::optional<AffineRelayOutputGateParams> output_gate;
};

struct RelayAffinePrefixInputs {
    Tensor packed;
    Tensor initial_state;
    Tensor actual_start;
    // L1, height-sharded one head per head core and kept alive across calls: the predecessor writes a head's entry
    // state into entry_landing, this rank's next state is staged in state_staging, and the last rank's final state
    // lands in the first rank's final_landing.
    Tensor entry_landing;
    Tensor state_staging;
    Tensor final_landing;
    std::optional<Tensor> gate_input;
    std::optional<Tensor> gate_weight;
    std::optional<Tensor> gate_copy_source;
};

// Legacy program descriptors, as select_final_carry's: the metal2 ProgramSpec cannot yet open fabric connections.
struct RelayAffinePrefixProgramFactory {
    // One program per coordinate: each sends to its own successor on the sequence-parallel ring.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const RelayAffinePrefixParams&,
        const RelayAffinePrefixInputs&,
        std::vector<Tensor>&,
        const ttnn::MeshCoordinateRangeSet&);
};

struct RelayAffinePrefixOperation {
    using operation_attributes_t = RelayAffinePrefixParams;
    using tensor_args_t = RelayAffinePrefixInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<RelayAffinePrefixProgramFactory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return RelayAffinePrefixProgramFactory{};
    }
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim
