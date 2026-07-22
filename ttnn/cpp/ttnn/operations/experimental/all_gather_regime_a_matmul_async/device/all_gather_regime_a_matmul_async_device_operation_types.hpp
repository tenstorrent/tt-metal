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
};

struct AllGatherRegimeAMatmulAsyncInputs {
    Tensor input_tensor;   // in0 : [.., M, K_local] (K-sharded across d devices), DRAM interleaved, bf16, TILE
    Tensor weight_tensor;  // in1 : [.., K_global, N], DRAM width-sharded (8 banks), bf16, TILE

    // CCL all-gather scaffolding (as in the existing AGMM).
    std::vector<GlobalSemaphore> multi_device_global_semaphore;
    std::optional<GlobalSemaphore> barrier_semaphore;
    std::optional<Tensor> persistent_output_buffer;
};

}  // namespace ttnn::experimental::prim
