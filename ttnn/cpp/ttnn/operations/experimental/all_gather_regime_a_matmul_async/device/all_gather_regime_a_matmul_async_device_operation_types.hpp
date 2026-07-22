// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_config.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include <tt-metalium/global_semaphore.hpp>

namespace ttnn::experimental::prim {

// Fused all-gather (of in0 on K) + regime_a_matmul. Task 2 scaffold: the D>1 streaming program factory is
// implemented in Task 3; this device op currently supports only the geometry/validation path.
struct AllGatherRegimeAMatmulAsyncParams {
    std::optional<RegimeAMatmulConfig> regime_a_config;

    // Multi-device / fabric geometry (mirrors the existing AGMM fabric arguments).
    uint32_t d = 1;                 // devices along the gather (cluster) axis
    std::optional<uint32_t> cluster_axis;
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring;
    uint32_t num_links = 1;
    uint32_t num_workers_per_link = 1;
    uint32_t num_buffers_per_channel = 2;

    // Transport-chunk / packet knobs (Task 3+; not public tuning knobs yet).
    uint32_t transport_c = 1;
    uint32_t transport_slots = 2;
    uint32_t packet_bytes = 4096;

    std::optional<tt::tt_metal::MemoryConfig> output_mem_config;
    std::optional<tt::tt_metal::DataType> output_dtype;
    DeviceComputeKernelConfig compute_kernel_config;

    // CCL semaphores live in operation_attributes (NOT tensor_args): the device_operation framework runs a
    // count_object_of_type<Tensor> traversal over tensor_args that would recurse into GlobalSemaphore's
    // CoreRangeSet (which has no reflection terminal) and throw. The attributes path hashes them safely (and
    // our custom compute_program_hash skips them entirely).
    std::vector<GlobalSemaphore> multi_device_global_semaphore;
    std::optional<GlobalSemaphore> barrier_semaphore;

    // Reflection surface (logging / default attribute visits) — DELIBERATELY excludes the GlobalSemaphores
    // (their CoreRangeSet has no reflection terminal and would throw "Unsupported visit"). The program-cache
    // key is our explicit compute_program_hash, so this list need not be exhaustive for caching.
    static constexpr auto attribute_names = std::make_tuple(
        "d",
        "cluster_axis",
        "topology",
        "num_links",
        "num_workers_per_link",
        "num_buffers_per_channel",
        "transport_c",
        "transport_slots",
        "packet_bytes",
        "regime_a_config",
        "output_mem_config",
        "output_dtype");

    auto attribute_values() const {
        return std::forward_as_tuple(
            this->d,
            this->cluster_axis,
            this->topology,
            this->num_links,
            this->num_workers_per_link,
            this->num_buffers_per_channel,
            this->transport_c,
            this->transport_slots,
            this->packet_bytes,
            this->regime_a_config,
            this->output_mem_config,
            this->output_dtype);
    }
};

struct AllGatherRegimeAMatmulAsyncInputs {
    Tensor input_tensor;   // in0 : [.., M, K_local] (K-sharded across d devices), DRAM interleaved, bf16, TILE
    Tensor weight_tensor;  // in1 : [.., K_global, N], DRAM width-sharded (8 banks), bf16, TILE

    // Only Tensors belong here (see the note on the semaphores above).
    std::optional<Tensor> persistent_output_buffer;
};

}  // namespace ttnn::experimental::prim
