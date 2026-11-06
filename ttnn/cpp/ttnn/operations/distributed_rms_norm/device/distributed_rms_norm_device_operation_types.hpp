// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::distributed_rms_norm {

// The cross-device schedule of the Σx² records. Every mode gives the same output.
enum class ExchangeMode : uint8_t { DIRECT, RELAY, MCAST };

// What the caller chose and the entry resolved. The solved plan is not an attribute: it is memoized in the
// planner (distributed_rms_norm_planner.hpp) and looked up by const&.
struct DistributedRmsNormParams {
    uint32_t cluster_axis = 0;
    // Runtime scalar (the compute kernel's EPS_BITS common arg): left out of the program key and re-applied by
    // override_runtime_arguments. Kept as the Python double; packed to fp32 where the Python packs it.
    double epsilon = 1e-6;
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear;
    uint32_t group_size = 0;                            // G, devices along cluster_axis
    uint32_t num_links = 0;                             // L resolved (None -> every usable link)
    bool ring = false;                                  // topology == Ring and G >= 3
    ExchangeMode exchange_mode = ExchangeMode::DIRECT;  // as asked (None -> DIRECT)
    // Resolved: the caller's or the default (HiFi4, fp32 DEST, exact math).
    tt::tt_metal::ComputeConfigDescriptor compute_config;
    tt::tt_metal::MemoryConfig output_memory_config;
};

struct DistributedRmsNormInputs {
    const Tensor& input;
    const std::optional<Tensor>& gamma;
    // Preallocated output (must match the spec the op would allocate).
    const std::optional<Tensor>& output;
    // The caller's exchange landing (ttnn::distributed_rms_norm_scratch_tensor); absent -> the call allocates it.
    const std::optional<Tensor>& scratch;
};

// Refusals of the support contract, named after ttnn.operations._op_contract.
struct SupportRefusal : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct UnsupportedAxisValue : SupportRefusal {
    using SupportRefusal::SupportRefusal;
};
// Reachable: {dtype=float32, fp32_dest_acc_en=False}.
struct ExcludedCell : SupportRefusal {
    using SupportRefusal::SupportRefusal;
};
// The Python's plain NotImplementedError (regime refusals: mcast on a 2D fabric, a sharded output without a
// shard spec, a shard layout or L1 fit the planner cannot build). The binding raises it as NotImplementedError.
struct RegimeRefusal : std::runtime_error {
    using std::runtime_error::runtime_error;
};
// Python ValueError: throw std::invalid_argument (nanobind maps it).

}  // namespace ttnn::operations::distributed_rms_norm
