// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct AgmmTunedParams {
    // The matmul's 2D multicast schedule over the gathered in0 (its grid leaves the gather a free worker column).
    operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig program_config;
    DeviceComputeKernelConfig compute_kernel_config;
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    // The mesh axis the input is sharded over (its K), and the fabric links per direction the gather uses.
    uint32_t cluster_axis;
    uint32_t num_links;
    // Persistent global semaphores (see agmm_tuned.hpp): on the gather's cores, and the slice counters
    // on the matmul's in0 senders (one per other rank, by its distance behind this one).
    uint32_t ready_semaphore_address;
    std::vector<uint32_t> counter_addresses;
};

struct AgmmTunedInputs {
    Tensor input;     // this rank's K shard of in0, [1, 1, M, K / ranks]
    Tensor weight;    // [1, 1, K, N]
    Tensor gathered;  // the gathered in0, [1, 1, M, K]: the other ranks' slices land here; persistent across calls
    // Each device's rows of what differs between the devices (agmm_tuned_gather_table_rows), so that one program
    // serves the mesh: [1, 1, rows, 64] UINT32 row-major in DRAM.
    Tensor gather_table;
};

}  // namespace ttnn::experimental::prim
