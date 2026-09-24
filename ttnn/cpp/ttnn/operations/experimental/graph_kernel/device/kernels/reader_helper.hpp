// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <utility>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Reader-side counterpart of chain_llk.hpp (normalization/layernorm_distributed).
//
// A Read_Node streams pages of one tensor into one dataflow buffer. chain_reads() takes a pack
// of node types and walks them as a sliding window: dfb_length pages of node 0, then dfb_length
// pages of node 1, ..., then back to node 0 for the next dfb_length pages, until every node has
// read all total_pages of this core's slice. A chain_llk compute kernel fed by these DFBs sees
// each node's buffer filled one dfb_length window at a time, in node order, which is exactly the
// order chain_llk consumes them.
//
// Everything that shapes the loop nest is a compile-time constant (total_pages, dfb_length,
// num_dst_regs, push_granularity), so window count, leftover window and padding are all folded.
// What is unrolled is one push: reserve, `push_granularity` NOC reads with constant entry offsets,
// barrier, push. The walk across pushes within a window and across windows is a plain runtime loop.
// Code size therefore grows with the number of nodes only, not with total_pages or dfb_length. This
// matters: a NOC read site costs roughly 450 B of binary and the per-core kernel budget is ~70 KB
// shared with compute and writer, so unrolling whole windows (dfb_length * nodes sites) overflowed
// at 16 inputs and unrolling everything overflowed at ~250 pages of 3 inputs. Only start_id, this
// core's first page, is a runtime value: a plain offset added to every page id.
//
// Push granularity is decoupled from the compute side's dest-register granularity:
//   * pages are pushed as they land, `push_granularity` at a time (default 1), so compute can
//     start on a window before the whole window has arrived;
//   * each node's push count per window is padded up to a multiple of num_dst_regs (4 for an
//     fp32 dest, 8 otherwise) with a blank reserve/push pair, so a consumer doing
//     wait_front(num_dst_regs) never stalls on an uneven tail. Blank entries hold stale L1;
//     the consumer must ignore the dest tiles they produce (chain_llk computes the same leftover
//     count from total_tiles / dfb_length, so it knows which ones).
//
// Usage, mirroring the LLK_Node pattern:
//
//   struct read_a_node {
//       static constexpr auto tensor = tensor::in0;
//       static constexpr Read_Node node{.DFB = dfb::a, .page_size = get_arg(args::page_size)};
//   };
//   struct read_b_node {
//       static constexpr auto tensor = tensor::in1;
//       static constexpr Read_Node node{.DFB = dfb::b, .page_size = get_arg(args::page_size)};
//   };
//
//   constexpr uint32_t pages_per_core = get_arg(args::num_pages);   // a CTA, not an RTA
//   chain_reads<pages_per_core, dfb_length, /*is_fp_32=*/true>(start_id, read_a_node{}, read_b_node{});
//
// start_id applies to every node: all tensors in a window share page indexing.

struct Read_Node {
    // Destination dataflow-buffer handle. Declared uint32_t so a dfb::<name> binding token
    // converts straight in at constexpr time.
    uint32_t DFB;
    // Bytes read from the tensor per page.
    uint32_t page_size;
    // Pages per reserve/read/barrier/push cycle. 1 pushes every page as soon as it lands.
    // Must divide the DFB's entry count so a reserve of this many entries is contiguous.
    uint32_t push_granularity = 1;
    // Debug mode: 0 off
    // Debug mode: 1 print every page after it lands
    uint32_t debug_mode = 0;
};

namespace graph_kernel_detail {

// Compile-time counted loop: calls f(std::integral_constant<uint32_t, I>{}) for I in [0, N).
// The body reads the index as `constexpr uint32_t I = decltype(i)::value;`. Bodies passed in are
// marked always_inline: at -O2 GCC otherwise outlines the larger ones as separate functions.
template <typename F, uint32_t... Is>
FORCE_INLINE void static_for_impl(F&& f, std::integer_sequence<uint32_t, Is...>) {
    (f(std::integral_constant<uint32_t, Is>{}), ...);
}

template <uint32_t N, typename F>
FORCE_INLINE void static_for(F&& f) {
    static_for_impl(f, std::make_integer_sequence<uint32_t, N>{});
}

}  // namespace graph_kernel_detail

template <uint32_t num_pages, uint32_t num_dst_regs, typename cur_read_type>
FORCE_INLINE void unroll_reads(uint32_t page_id_start);

template <uint32_t n, typename cur_read_type, typename Accessor>
FORCE_INLINE void read_push(
    const Noc& noc, const Accessor& accessor, DataflowBuffer& dfb, uint32_t first_page, uint32_t entry_size);

template <typename cur_read_type, uint32_t num_pages>
FORCE_INLINE void print_landed_pages(DataflowBuffer& dfb);

template <uint32_t total_pages, uint32_t dfb_length, bool is_fp_32, typename... read_nodes>
FORCE_INLINE void chain_reads(uint32_t start_id, read_nodes...) {
    constexpr uint32_t num_dst_regs = (is_fp_32 ? 4 : 8);
    static_assert(dfb_length % num_dst_regs == 0, "graph_kernel: dfb_length must be a multiple of num_dst_regs");

    constexpr uint32_t iterations = total_pages / dfb_length;
    constexpr uint32_t leftovers = total_pages % dfb_length;

    // Sliding window: every node reads its slice of window i before any node starts window i+1.
    // Each node's window body is unrolled; the window loop itself is not (see the header comment).
    for (uint32_t i = 0; i < iterations; ++i) {
        const uint32_t window_start = start_id + i * dfb_length;
        (..., unroll_reads<dfb_length, num_dst_regs, read_nodes>(window_start));
    }
    if constexpr (leftovers != 0) {
        (..., unroll_reads<leftovers, num_dst_regs, read_nodes>(start_id + iterations * dfb_length));
    }
}

// One node, one window: stream num_pages pages into the node's DFB, `push_granularity` at a time,
// then pad the push count up to a multiple of num_dst_regs. For a full window
// (num_pages == dfb_length) the pad is zero by the static_assert above; only the leftover window
// ever pads. The full pushes run in a runtime loop; the uneven tail push is a separate, constant
// instantiation.
template <uint32_t num_pages, uint32_t num_dst_regs, typename cur_read_type>
FORCE_INLINE void unroll_reads(uint32_t page_id_start) {
    constexpr auto cur_read = cur_read_type::node;
    constexpr uint32_t granularity = cur_read.push_granularity;
    static_assert(granularity != 0, "graph_kernel: push_granularity must be at least 1");
    constexpr uint32_t full_pushes = num_pages / granularity;
    constexpr uint32_t tail = num_pages % granularity;
    constexpr uint32_t pad = (num_dst_regs - (num_pages % num_dst_regs)) % num_dst_regs;

    DataflowBuffer dfb(cur_read.DFB);
    Noc noc;
    const auto accessor = TensorAccessor(cur_read_type::tensor);
    const uint32_t entry_size = dfb.get_entry_size();

    for (uint32_t p = 0; p < full_pushes; ++p) {
        read_push<granularity, cur_read_type>(noc, accessor, dfb, page_id_start + p * granularity, entry_size);
    }
    if constexpr (tail != 0) {
        read_push<tail, cur_read_type>(noc, accessor, dfb, page_id_start + full_pushes * granularity, entry_size);
    }

    if constexpr (pad != 0) {
        // Blank entries: no read, just advance the ring so the consumer's wait_front(num_dst_regs) clears.
        dfb.reserve_back(pad);
        dfb.push_back(pad);
    }
}

// One push: reserve n entries, issue n NOC reads into consecutive entries (unrolled, constant
// offsets), wait for them to land, push them.
template <uint32_t n, typename cur_read_type, typename Accessor>
FORCE_INLINE void read_push(
    const Noc& noc, const Accessor& accessor, DataflowBuffer& dfb, uint32_t first_page, uint32_t entry_size) {
    constexpr auto cur_read = cur_read_type::node;
    dfb.reserve_back(n);
    graph_kernel_detail::static_for<n>([&](auto k) __attribute__((always_inline)) {
        constexpr uint32_t K = decltype(k)::value;
        noc.async_read(
            accessor, dfb, cur_read.page_size, {.page_id = first_page + K}, {.offset_bytes = K * entry_size});
    });
    noc.async_read_barrier();
    if constexpr (cur_read.debug_mode == 1) {
        print_landed_pages<cur_read_type, n>(dfb);
    }
    dfb.push_back(n);
}

template <typename cur_read_type, uint32_t num_pages>
FORCE_INLINE void print_landed_pages(DataflowBuffer& dfb) {
    [[maybe_unused]] constexpr auto cur_read = cur_read_type::node;
    // Commented out so code will compile on non debug print modes. Uncomment for debug purposes.
    // DPRINT << "=============DFB " << cur_read.DFB << " landed " << num_pages << " pages=============" << ENDL();
    // print_pages(dfb.get_write_ptr(), cur_read.page_size / sizeof(uint16_t), num_pages);
}
