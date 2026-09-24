// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#include "chain_llk.hpp"
#include "input_bindings.hpp"

// graph_kernel compute: the chain_llk side of the read_nodes. The host parses the expression text
// into binary nodes and hands them over as the GK_GRAPH define, four numbers per node:
//
//     op_a_b_out[_op_a_b_out...]      e.g. "a*b+c"  ->  2_0_1_128_0_128_2_255
//
//   op      0 add, 1 sub, 2 mul (index into the op table below)
//   operand   0..127  input i        (dfb::in<i>, filled by the reader)
//             128..254 intermediate k (dfb::t<k>, produced by an earlier node)
//             255      the output     (dfb::out, drained by the writer)
//
// Every node runs as one LLK_Node with A and B consumed in lockstep (fixed_DFB_B_index = 0xFFFF).

// ---- Op table -------------------------------------------------------------------------------------
//
// noinline: chain_llk calls these through a constexpr function pointer, which GCC would otherwise
// inline into every node's unrolled dest-register loop. With 25 nodes that duplicated the eltwise
// LLK body 200 times and pushed the three TRISC binaries to ~60 KB, past the per-core kernel budget.
// One body per op keeps the compute kernel a few KB regardless of the expression size.

template <uint32_t op>
__attribute__((noinline)) void binary_init(uint32_t icb0, uint32_t icb1, uint32_t call_line) {
    if constexpr (op == 0) {
        add_init(icb0, icb1, false, call_line);
    } else if constexpr (op == 1) {
        sub_init(icb0, icb1, false, call_line);
    } else {
        mul_init(icb0, icb1, true, call_line);
    }
}

template <uint32_t op>
__attribute__((noinline)) void binary_compute(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    if constexpr (op == 0) {
        add_tiles(icb0, icb1, itile0, itile1, idst);
    } else if constexpr (op == 1) {
        sub_tiles(icb0, icb1, itile0, itile1, idst);
    } else {
        mul_tiles(icb0, icb1, itile0, itile1, idst);
    }
}

// ---- Graph decoding ---------------------------------------------------------------------------------

#define GK_STR2(x) #x
#define GK_STR(x) GK_STR2(x)
constexpr const char* graph_str = GK_STR(GK_GRAPH);

constexpr size_t count_fields(const char* s) {
    size_t n = 1;
    for (; *s != '\0'; ++s) {
        n += (*s == '_');
    }
    return n;
}

struct NodeSpec {
    uint32_t op, a, b, out;
};

template <size_t num_nodes>
struct Graph {
    NodeSpec nodes[num_nodes];
};

constexpr void set_field(NodeSpec& node, size_t field, uint32_t value) {
    switch (field) {
        case 0: node.op = value; break;
        case 1: node.a = value; break;
        case 2: node.b = value; break;
        default: node.out = value; break;
    }
}

template <size_t num_nodes>
constexpr Graph<num_nodes> parse_graph(const char* s) {
    Graph<num_nodes> g{};
    size_t field = 0;
    uint32_t value = 0;
    for (;; ++s) {
        if (*s == '_' || *s == '\0') {
            set_field(g.nodes[field / 4], field % 4, value);
            ++field;
            value = 0;
            if (*s == '\0') {
                break;
            }
        } else {
            value = value * 10 + static_cast<uint32_t>(*s - '0');
        }
    }
    return g;
}

constexpr size_t num_nodes = count_fields(graph_str) / 4;
static_assert(count_fields(graph_str) % 4 == 0, "graph_kernel: GK_GRAPH must hold four fields per node");
constexpr Graph<num_nodes> graph = parse_graph<num_nodes>(graph_str);

template <uint32_t code>
constexpr uint32_t operand_dfb() {
    if constexpr (code < 128) {
        return graph_kernel::input_dfb<code>();
    } else if constexpr (code == 255) {
        return dfb::out;
    } else {
        return graph_kernel::interm_dfb<code - 128>();
    }
}

// ---- Nodes ------------------------------------------------------------------------------------------

template <size_t I>
struct graph_node {
    static constexpr NodeSpec spec = graph.nodes[I];
    static constexpr LLK_Node node{
        .llk_init = &binary_init<spec.op>,
        .llk = FN_compute(static_cast<fn_compute_5*>(&binary_compute<spec.op>)),
        .DFB_A = operand_dfb<spec.a>(),
        .DFB_B = operand_dfb<spec.b>(),
        .DFB_OUT = operand_dfb<spec.out>(),
        .fixed_DFB_B_index = 0xFFFF,
        .fixed_dest_reg = 0xFFFF,
        .debug_mode = 0,
    };
};

template <uint32_t pages_per_core, uint32_t dfb_length, bool is_fp_32, size_t... Is>
void run_graph(std::index_sequence<Is...>) {
    chain_llk<pages_per_core, dfb_length, is_fp_32>(graph_node<Is>{}...);
}

void kernel_main() {
    constexpr uint32_t pages_per_core = get_arg(args::pages_per_core);
    constexpr uint32_t dfb_length = get_arg(args::dfb_length);
    constexpr bool is_fp_32 = get_arg(args::is_fp_32) != 0;

    compute_kernel_hw_startup(graph_node<0>::node.DFB_A, graph_node<0>::node.DFB_B, dfb::out);
    run_graph<pages_per_core, dfb_length, is_fp_32>(std::make_index_sequence<num_nodes>{});
}
