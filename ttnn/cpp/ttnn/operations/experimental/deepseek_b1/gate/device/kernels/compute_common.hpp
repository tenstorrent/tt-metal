// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"

/*

    1. Router: Matmul: [1, 7168] @ [7168, 256] -> [1, 256]
    2. Sigmoid [1, 256]
    3. Add expert bias [1, 256]
    4. Top-8 operation: [1, 256] -> [1, 8]
    ...
    5. Normalize over [1, 8]

*/

// cb_in0: activations [1, 7168]
// cb_in1:  router weights [7168, 256]
// cb_out: router scores [1, 256]
template <uint32_t cb_in0, uint32_t cb_in1, uint32_t cb_out>
void router_compute() {}
