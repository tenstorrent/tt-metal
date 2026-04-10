// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/matmul_helpers_compute.hpp"

using std::uint32_t;
using namespace compute_kernel_lib;

// matmul C=A*B using dims MK*KN = MN (row major order)
//
void kernel_main() {
    uint32_t batch = get_compile_time_arg_val(0);
    uint32_t Mt = get_compile_time_arg_val(1);
    uint32_t Kt = get_compile_time_arg_val(2);
    uint32_t Nt = get_compile_time_arg_val(3);

    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");

    auto cfg = MatmulConfig::tile(cb_in0, cb_in1, cb_out);
    matmul_init<TILE>(cfg);
    matmul<TILE>(cfg, MatmulBlockShape::of(batch, Mt, Nt, Kt, 1, 1, 1, 1, 1));
}
