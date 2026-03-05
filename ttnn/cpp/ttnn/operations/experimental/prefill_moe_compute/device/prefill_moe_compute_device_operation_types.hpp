// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::prefill_moe_compute {

struct operation_attributes_t {
    uint32_t num_experts;  // Number of experts to compute (1-4)
    uint32_t num_cores;    // Number of matmul cores (grid_x * grid_y)
    uint32_t grid_x;       // Matmul core grid X dimension
    uint32_t grid_y;       // Matmul core grid Y dimension
    // Per-device combine routing metadata.
    // Index = linearized mesh coordinate (row-major).
    // Single device: vector of size 1.  Multi-device: one entry per device.
    // Each inner vector is packed as:
    //   Per expert: [out_buf_addr, M_e, token_row[0..M_e-1], weight_bf16[0..M_e-1]]
    std::vector<std::vector<uint32_t>> per_device_combine_metadata;
    // When true, use TT-Fabric to all-reduce partial outputs across mesh devices.
    // Requires reduce_recv_buf in tensor_args.
    bool enable_fabric_reduce = false;
    // When true, use TT-Fabric dispatch kernel for token exchange between devices.
    // Requires hidden_states_rm, staging_buf, and dispatch_metadata.
    bool enable_fabric_dispatch = false;
    // Per-device dispatch routing metadata.
    // Each inner vector is packed as:
    //   [local_count, recv_count, send_count, local_indices..., send_indices...]
    std::vector<std::vector<uint32_t>> dispatch_metadata;
};

struct tensor_args_t {
    const Tensor& hidden_states;                 // [1,1,P,D] BF16 input activation
    const Tensor& pkt_buf;                       // [1,1,P,D] BF16 scratch for dispatch
    const Tensor& inter_buf;                     // [1,1,P,D] BF16 scratch for intermediate
    const Tensor& output;                        // [1,1,P,D] BF16 pre-allocated output (zero-filled)
    const std::vector<Tensor>& gate_up_weights;  // Per-expert [1,1,D,D_FF] BFP4_b
    const std::vector<Tensor>& down_weights;     // Per-expert [1,1,D_FF/2,D] BFP4_b
    const std::vector<Tensor>& out_bufs;         // Per-expert [1,1,P,D] BF16 scratch
    std::optional<Tensor> reduce_recv_buf;       // [1,1,P,D] BF16 receive buffer for fabric reduce
    std::optional<Tensor> hidden_states_rm;      // [1,1,P,D] BF16 ROW_MAJOR for fabric dispatch
    std::optional<Tensor> staging_buf;           // [1,1,P,D] BF16 ROW_MAJOR fabric receive staging
};

// Output: the pre-allocated output tensor (modified in-place)
using tensor_return_value_t = Tensor;
using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::prefill_moe_compute
