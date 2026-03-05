// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel
// Reads RM input sticks from DRAM, provides scaler/epsilon/gamma/beta tiles.
//
// Runtime args (per core):
//   [0] src_addr         - Input buffer base address
//   [1] num_blocks       - Number of tile-rows for this core
//   [2] start_stick_id   - First stick index (row index)
//   [3] gamma_addr       - Gamma buffer base address (0 if no gamma)
//   [4] beta_addr        - Beta buffer base address (0 if no beta)
//   [5] eps_value        - Epsilon as bit-cast uint32
//   [6] mean_scaler_value - 1/W as bit-cast uint32
//
// Compile-time args:
//   [0] stick_size       - W * element_size bytes
//   [1+] TensorAccessorArgs(input)
//
// Real implementation will use:
//   #include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
//   #include "api/tensor/tensor_accessor_args.h"

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Stub: Real implementation reads RM sticks and produces scaler/eps/gamma/beta tiles
    // TODO: implement in kernel-writer stage
}
