// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <utility>

#include "input_bindings.hpp"
#include "reader_helper.hpp"

// graph_kernel reader: input I (tensor in<I>) is streamed into dataflow buffer in<I> by chain_reads,
// as a sliding window over all inputs. The node list is expanded from the num_inputs compile-time
// arg, so there are no per-count cases.

template <size_t I>
struct read_node {
    static constexpr auto tensor = graph_kernel::input_tensor<I>();
    static constexpr Read_Node node{.DFB = graph_kernel::input_dfb<I>(), .page_size = get_arg(args::page_size)};
};

template <uint32_t pages_per_core, uint32_t dfb_length, bool is_fp_32, size_t... Is>
FORCE_INLINE void read_all_inputs(uint32_t start_id, std::index_sequence<Is...>) {
    chain_reads<pages_per_core, dfb_length, is_fp_32>(start_id, read_node<Is>{}...);
}

void kernel_main() {
    constexpr uint32_t num_inputs = get_arg(args::num_inputs);
    constexpr uint32_t pages_per_core = get_arg(args::pages_per_core);
    constexpr uint32_t dfb_length = get_arg(args::dfb_length);
    constexpr bool is_fp_32 = get_arg(args::is_fp_32) != 0;
    const uint32_t start_id = get_arg(args::start_id);

    read_all_inputs<pages_per_core, dfb_length, is_fp_32>(start_id, std::make_index_sequence<num_inputs>{});
}
