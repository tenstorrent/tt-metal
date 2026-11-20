// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/relay_chunk_scan/device/kernels/dataflow/gate_matmul_dataflow.hpp"

constexpr uint32_t cb_gate = get_compile_time_arg_val(0);
constexpr uint32_t num_chunks = get_compile_time_arg_val(1);
constexpr uint32_t cores_per_head = get_compile_time_arg_val(2);
// What this kernel moves: x (on RISC 1, NoC 1), W (on RISC 0, NoC 0), or both where a sender leaves the helper one
// RISC.
constexpr bool moves_x = get_compile_time_arg_val(3);
constexpr bool moves_w = get_compile_time_arg_val(4);
// The output gate's operands (gate_matmul_dataflow.hpp), whose args follow the landed count in the common args.
using Gate = kda_gate::Operands<5>;

// A head's helper forms g for the chunks of the head's first core, which runs the relay's hop instead: it takes the
// rows of x of those chunks from the group of the helpers of the head's batch and the head's columns of W from the
// group of the head's cores (gate_matmul_dataflow.hpp), swaps partial sums with the helper of the pair's other head,
// then writes the g compute forms into the head core's cb_gate (at the same address) and counts it there, on the NoC
// that runs towards the head core.
void kernel_main() {
    const uint32_t head = get_arg_val<uint32_t>(0);
    const uint32_t head_xy = get_arg_val<uint32_t>(1);
    // The head core is below a helper whose head's W moves north, above one whose W moves south.
    const bool north = get_arg_val<uint32_t>(2);
    const uint32_t pair_xy = get_arg_val<uint32_t>(3);
    const uint32_t x_group = Gate::ways * (head / Gate::heads * cores_per_head) + Gate::half(head);
    const uint32_t first_row = head / Gate::heads * num_chunks;
    if constexpr (moves_x && moves_w) {
        // Each operand as far as its group lets it.
        auto x = Gate::x_share(x_group);
        auto w = Gate::w_share(head);
        const auto read_x = Gate::x_read(first_row, Gate::half(head));
        const auto read_w = Gate::w_read(head);
        bool x_done = false;
        bool w_done = false;
        while (!(x_done && w_done)) {
            x_done = x.step(Gate::blocks, read_x);
            w_done = w.step(Gate::blocks, read_w);
        }
        x.finish(Gate::blocks);
        w.finish(Gate::blocks);
    } else if constexpr (moves_x) {
        Gate::move_x(head / Gate::heads * cores_per_head, first_row, head);
        if (north) {
            return;
        }
    } else {
        Gate::move_w(head);
        if (!north) {
            return;
        }
    }

    if constexpr (Gate::ways == 2) {
        Gate::swap(pair_xy);
    }
    cb_wait_front(cb_gate, Gate::rows * Gate::columns);
    const uint32_t gate = get_read_ptr(cb_gate);
    noc_async_write(gate, kda_gate::noc_address(head_xy, gate), Gate::rows * Gate::columns * get_tile_size(cb_gate));
    noc_async_write_barrier();
    noc_semaphore_inc(kda_gate::noc_address(head_xy, get_common_arg_val<uint32_t>(0)), 1);
    cb_pop_front(cb_gate, Gate::rows * Gate::columns);
    noc_async_atomic_barrier();
}
