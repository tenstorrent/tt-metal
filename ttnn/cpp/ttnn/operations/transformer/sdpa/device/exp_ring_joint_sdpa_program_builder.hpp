// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared exp ring joint SDPA program construction: SDPA/fabric-MUX grid split, head-serial passes, row
// multicast chains, fused K/V all-gather over the MUX, runtime-argument layout and the dataflow CBs. The
// compute contract is supplied by a ComputeVariant: ExpRingJointSDPAProgramFactory (legacy exp-ring compute,
// including FAST) and ExpRingJointSDPARecipeProgramFactory (named precision recipes B-E) each own one.
// Defined in exp_ring_joint_sdpa_program_factory.cpp next to the legacy factory.

#include <cstdint>
#include <map>
#include <optional>
#include <string>

#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_device_operation_types.hpp"

namespace ttnn::prim::exp_ring_joint_sdpa {

struct KernelSources {
    std::string reader;
    std::string writer;  // Both the plain and the fabric-MUX (USE_MUX) writer.
    std::string compute;
};

class ComputeVariant {
public:
    // Matmul subblock width of fixed compute schedules.
    static constexpr uint32_t kFixedSubblockW = 4;

    virtual ~ComputeVariant() = default;

    virtual KernelSources kernel_sources() const = 0;

    // Called once per program with the resolved compute kernel configuration.
    virtual void configure(
        const ExpRingJointSDPAParams& /*args*/,
        const ExpRingJointSDPAInputs& /*tensor_args*/,
        bool /*fp32_dest_acc_en*/) {}

    // Fixed QK/PV matmul subblock height of the compute schedule, which then streams and accepts a partial
    // last Q row group; nullopt lets the host choose.
    virtual std::optional<uint32_t> fixed_subblock_h() const { return std::nullopt; }

    // Replaces the exp-ring CBs in desc.cbs with the variant's own compute layout and adds its defines.
    // Returns false (leaving desc and defines untouched) to keep the exp-ring layout.
    virtual bool replace_cbs(
        tt::tt_metal::ProgramDescriptor& /*desc*/,
        const tt::tt_metal::CoreRangeSet& /*sdpa_grid*/,
        uint32_t /*Sq_chunk_t*/,
        uint32_t /*DHt*/,
        std::map<std::string, std::string>& /*defines*/) {
        return false;
    }

    // Rejects a variant-owned layout that does not fit L1 (only called when replace_cbs returned true).
    virtual void check_l1(uint64_t /*cb_bytes*/, uint64_t /*usable_l1*/, uint32_t /*q_chunk_size*/) const {}

    virtual std::optional<tt::tt_metal::KernelDescriptor::ConfigDescriptor> compute_config() const {
        return std::nullopt;
    }
};

tt::tt_metal::WorkloadDescriptor build_workload_descriptor(
    const ExpRingJointSDPAParams& operation_attributes,
    const ExpRingJointSDPAInputs& tensor_args,
    ExpRingJointSDPAResult& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    ComputeVariant& variant);

// Re-patches the hash-excluded per-link GlobalSemaphore runtime args on a cache hit.
void apply_semaphore_runtime_args(
    tt::tt_metal::distributed::MeshWorkload& workload,
    const ExpRingJointSDPAParams& operation_attributes,
    const ExpRingJointSDPAInputs& tensor_args);

}  // namespace ttnn::prim::exp_ring_joint_sdpa
