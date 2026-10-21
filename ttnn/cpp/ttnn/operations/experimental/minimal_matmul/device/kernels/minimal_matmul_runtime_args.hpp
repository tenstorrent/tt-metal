// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Common runtime-argument slots of the legacy minimal_matmul kernels, shared by the host factory
// (minimal_matmul_program_factory.cpp) and the kernels. M_tiles is the one per-call length word;
// every per-core row range is derived from it on device (runtime_length.hpp).
namespace ttnn::experimental::prim::minimal_matmul_common_args {

// in0 and in1 data-movement kernels (sender and receiver flavours).
struct Dm {
    static constexpr uint32_t M_tiles = 0;
    static constexpr uint32_t in_addr = 1;  // in0 on the in0 kernels, in1 on the in1 kernels
    static constexpr uint32_t in2_addr = 2;
    static constexpr uint32_t in3_addr = 3;  // in0's second source; 0 on the in1 kernels
    static constexpr uint32_t ternary_a_addr = 4;
    static constexpr uint32_t ternary_b_addr = 5;
    static constexpr uint32_t out_addr = 6;  // N_chunks output addresses
};

struct Compute {
    static constexpr uint32_t M_tiles = 0;
    static constexpr uint32_t count = 1;
};

}  // namespace ttnn::experimental::prim::minimal_matmul_common_args
