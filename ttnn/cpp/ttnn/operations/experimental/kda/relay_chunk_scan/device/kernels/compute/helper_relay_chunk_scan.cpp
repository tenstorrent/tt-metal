// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/relay_chunk_scan/device/kernels/compute/gate_matmul.hpp"

constexpr uint32_t cb_gin0 = get_compile_time_arg_val(0);
constexpr uint32_t cb_gin1 = get_compile_time_arg_val(1);
constexpr uint32_t cb_gate = get_compile_time_arg_val(2);
constexpr uint32_t rows = get_compile_time_arg_val(3);
constexpr uint32_t columns = get_compile_time_arg_val(4);
constexpr uint32_t gate_kb = get_compile_time_arg_val(5);
constexpr uint32_t blocks = get_compile_time_arg_val(6);
constexpr auto gate_fidelity = static_cast<ckernel::MathFidelity>(get_compile_time_arg_val(7));
constexpr uint32_t cb_gate_pair = get_compile_time_arg_val(8);
constexpr uint32_t cb_gate_partial = get_compile_time_arg_val(9);

// A head's helper: g of the chunks of the head's first core (see the helper's data movement), its head's partial
// sums with the pair's helper's added.
void kernel_main() {
    const bool own_first = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_gin0, cb_gin1, cb_gate);
    DeviceZoneScopedN("HELPER-MM");
    kda_gate::matmul<gate_fidelity>(cb_gin0, cb_gin1, cb_gate, cb_gate_pair, own_first, rows, columns, gate_kb, blocks);
    kda_gate::add_pair(cb_gate_partial, cb_gate, rows * columns);
}
