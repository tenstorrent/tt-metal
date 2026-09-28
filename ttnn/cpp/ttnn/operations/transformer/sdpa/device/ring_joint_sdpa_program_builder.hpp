// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared ring joint SDPA program construction: ring/all-gather transport, core work split, K/V
// store-and-forward chains, runtime-argument layout and the dataflow CBs. The compute contract is supplied by
// a ComputeVariant: RingJointSDPAProgramFactory (legacy compute, including FAST) and
// RingJointSDPARecipeProgramFactory (named precision recipes B-E) each own one.
// Defined in ring_joint_sdpa_program_factory.cpp next to the legacy factory.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation_types.hpp"

namespace ttnn::prim::ring_joint_sdpa {

struct KernelSources {
    std::string reader;
    std::string writer;
    std::string compute;
};

// Compute-visible CB roles a variant may pin to fixed indices.
enum class ComputeCb {
    Q,
    K,
    V,
    ReduceScaler,
    ColumnIdentity,
    RecipScratch,
    QkIm,
    OutImA,
    OutImB,
    MaxA,
    MaxB,
    SumA,
    SumB,
    ExpMaxDiff,
    Out,
};

class ComputeVariant {
public:
    virtual ~ComputeVariant() = default;

    virtual KernelSources kernel_sources() const = 0;

    // Called once per program after chunk geometry and grid are known and before any dataflow CB is
    // allocated. A variant may seed desc.cbs with its own fixed-index CBs.
    virtual void configure(
        const RingJointSDPAParams& /*args*/,
        const RingJointSDPAInputs& /*tensor_args*/,
        tt::tt_metal::ProgramDescriptor& /*desc*/,
        const tt::tt_metal::CoreRangeSet& /*grid*/,
        uint32_t /*Sq_chunk_t*/,
        uint32_t /*Sk_chunk_t*/,
        uint32_t /*DHt*/) {}

    // Fixed QK/PV matmul subblock height of the compute schedule, which then also accepts a partial last Q
    // row group; nullopt lets the host choose (Sq must then divide by it).
    virtual std::optional<uint32_t> fixed_subblock_h(bool /*fp32_dest_acc_en*/) const { return std::nullopt; }
    // Matmul subblock width of a fixed compute schedule for a QK (K chunk) or PV (head dim) product of `tiles`
    // output columns; only consulted when fixed_subblock_h is set.
    virtual uint32_t fixed_subblock_w(uint32_t tiles) const { return tiles < 4 ? tiles : 4; }

    // Compute continues one recurrent state across every active ring iteration and normalizes only on the
    // last: streaming compute, no LSE / accumulator DRAM staging.
    virtual bool resident_ring_state() const { return false; }

    // First CB index for the ring dataflow CBs (above any fixed compute layout).
    virtual uint32_t first_dataflow_cb_index() const { return 0; }
    virtual std::optional<uint32_t> fixed_cb_index(ComputeCb /*role*/) const { return std::nullopt; }

    // Called after every ring CB is in desc.cbs (L1 fitting and any extra variant CBs).
    virtual void finalize_cbs(
        tt::tt_metal::ProgramDescriptor& /*desc*/,
        tt::tt_metal::IDevice* /*device*/,
        uint32_t /*cb_q_in*/,
        uint32_t /*q_chunk_bytes*/) const {}

    virtual void append_writer_compile_time_args(std::vector<uint32_t>& /*args*/, const RingJointSDPAResult&) const {}
    // Runs before the legacy writer common runtime args, so a variant's common args start at index 0.
    virtual void append_writer_common_runtime_args(
        tt::tt_metal::KernelDescriptor& /*writer*/, const RingJointSDPAResult&) const {}
    virtual void append_defines(tt::tt_metal::KernelDescriptor::Defines& /*defines*/) const {}
    virtual std::optional<tt::tt_metal::KernelDescriptor::ConfigDescriptor> compute_config() const {
        return std::nullopt;
    }
};

tt::tt_metal::WorkloadDescriptor build_workload_descriptor(
    const RingJointSDPAParams& args,
    const RingJointSDPAInputs& tensor_args,
    RingJointSDPAResult& output_tensors,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    ComputeVariant& variant);

// Re-applies the per-dispatch scalar runtime args (indexed KV cache / KV-pad rotation) on a cache hit.
void apply_scalar_runtime_args(
    tt::tt_metal::Program& program,
    const RingJointSDPAParams& args,
    const RingJointSDPAInputs& tensor_args,
    const ttnn::MeshCoordinate& coord);

}  // namespace ttnn::prim::ring_joint_sdpa
