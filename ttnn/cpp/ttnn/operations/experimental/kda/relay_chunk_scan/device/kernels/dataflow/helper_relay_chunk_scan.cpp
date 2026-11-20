// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/persistent_counter.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/relay_chunk_scan/device/kernels/dataflow/gate_matmul_dataflow.hpp"

constexpr uint32_t cb_gate = get_compile_time_arg_val(0);
constexpr uint32_t num_chunks = get_compile_time_arg_val(1);
constexpr uint32_t cores_per_head = get_compile_time_arg_val(2);
// What this kernel moves: x (on RISC 1, NoC 1), W (on RISC 0, NoC 0), or both where a sender leaves the helper one
// RISC.
constexpr bool moves_x = get_compile_time_arg_val(3);
constexpr bool moves_w = get_compile_time_arg_val(4);
// In place, the head core's factors (compute forms them from g and the norm weight), else its g.
constexpr bool streamed = get_compile_time_arg_val(5);
constexpr uint32_t cb_gate_factor = get_compile_time_arg_val(6);
constexpr uint32_t cb_weight = get_compile_time_arg_val(7);
constexpr auto weight_args = TensorAccessorArgs<8>();
// The output gate's operands (gate_matmul_dataflow.hpp), whose args follow the landed count and the norm weight in the
// common args.
using Gate = kda_gate::Operands<weight_args.next_compile_time_args_offset()>;

// A head's helper forms g for the chunks of the head's first core, which runs the relay's hop instead: it takes the
// rows of x of those chunks from the group of the helpers of the head's batch and the head's columns of W from the
// group of the head's cores (gate_matmul_dataflow.hpp), swaps partial sums with the helper of the pair's other head,
// then writes what compute forms into the head core's buffer at the same address (in place its factors, once the head
// core has counted here that their storage is free; streamed its g) and counts it there, on the NoC that runs towards
// the head core.
void kernel_main() {
    const uint32_t head = get_arg_val<uint32_t>(0);
    const uint32_t head_xy = get_arg_val<uint32_t>(1);
    // The head core is below a helper whose head's W moves north, above one whose W moves south.
    const bool north = get_arg_val<uint32_t>(2);
    const uint32_t pair_xy = get_arg_val<uint32_t>(3);
    // The kernel that writes to the head core reads the norm weight for compute meanwhile.
    const bool writes = moves_x && moves_w ? true : moves_x ? !north : north;
    if (!streamed && writes) {
        const auto weight = TensorAccessor(weight_args, get_common_arg_val<uint32_t>(1));
        cb_reserve_back(cb_weight, Gate::columns);
        for (uint32_t j = 0; j < Gate::columns; ++j) {
            noc_async_read_page(j, weight, get_write_ptr(cb_weight) + j * get_tile_size(cb_weight));
        }
    }
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
    const uint32_t landed = get_common_arg_val<uint32_t>(0);
    constexpr uint32_t cb_out = streamed ? cb_gate : cb_gate_factor;
    if constexpr (!streamed) {
        noc_async_read_barrier();
        cb_push_back(cb_weight, Gate::columns);
        kda_counter::wait_and_consume(landed, 1);
    }
    cb_wait_front(cb_out, Gate::rows * Gate::columns);
    const uint32_t out = get_read_ptr(cb_out);
    noc_async_write(out, kda_gate::noc_address(head_xy, out), Gate::rows * Gate::columns * get_tile_size(cb_out));
    noc_async_write_barrier();
    noc_semaphore_inc(kda_gate::noc_address(head_xy, landed), 1);
    cb_pop_front(cb_out, Gate::rows * Gate::columns);
    noc_async_atomic_barrier();
}
