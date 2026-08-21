// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implementation file for local_copy_helpers_dataflow.hpp
// Do not include directly - include local_copy_helpers_dataflow.hpp instead

#pragma once

namespace dataflow_kernel_lib {

// ─── local_addr ─────────────────────────────────────────────────────────────
//
// A self-aimed unicast source. Every helper below routes its source through this, so the
// per-NoC coordinate rule is enforced in exactly one place.
//
// Example — a copy too large for the stateful path, issued directly:
//
//   #include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
//   Noc noc;
//   UnicastEndpoint self_ep;
//   noc.async_read(
//       self_ep,
//       dst_dfb,
//       n_bytes,
//       dataflow_kernel_lib::local_addr(src_l1_addr, noc.get_noc_id()),
//       {.offset_bytes = dst_offset});
//   noc.async_read_barrier();
//
FORCE_INLINE noc_traits_t<UnicastEndpoint>::src_args_type local_addr(uint32_t addr, uint8_t noc_id) {
    // my_x / my_y are indexed by NoC id on purpose: NOC 0 and NOC 1 have different coordinate
    // spaces, so my_x[0] is the WRONG x for a transfer issued on NOC 1.
    return noc_traits_t<UnicastEndpoint>::src_args_type{.noc_x = my_x[noc_id], .noc_y = my_y[noc_id], .addr = addr};
}

// ─── set_read_state / read_with_state ───────────────────────────────────────
//
// The stateful split exists to amortise NoC config-register setup across a loop: program the
// source address and the size ONCE, then issue N reads that only rewrite the destination.
//
// Example — fill a destination region with N copies of one padding stick:
//
//   Noc noc;
//   dataflow_kernel_lib::set_read_state<stick_bytes>(noc, pad_l1_addr);
//   uint32_t dst = dst_base;
//   for (uint32_t k = 0; k < nsticks; ++k) {
//       dataflow_kernel_lib::read_with_state(noc, dst, pad_l1_addr);
//       dst += stick_bytes;
//   }
//   noc.async_read_barrier();
//
// Example — scatter into a typed destination (DataflowBuffer / CircularBuffer):
//
//   dataflow_kernel_lib::set_read_state<row_bytes>(noc, src_l1_base);
//   for (uint32_t r = 0; r < nrows; ++r) {
//       dataflow_kernel_lib::read_with_state(noc, dst_dfb, src_l1_base, {.offset_bytes = r * row_bytes});
//   }
//   noc.async_read_barrier();
//
// Changing the source address or the size means calling set_read_state() again.
//
template <uint32_t transfer_size>
FORCE_INLINE void set_read_state(Noc noc, uint32_t src_addr) {
    // Single-packet bound: the stateful path programs one packet's worth of state, so a
    // multi-packet transfer cannot be expressed here and must use noc.async_read instead.
    static_assert(transfer_size <= NOC_MAX_BURST_SIZE, "Use noc.async_read for multi-packet transfers");
    UnicastEndpoint ep;
    noc.set_async_read_state<NocOptions::DEFAULT, transfer_size>(
        ep, transfer_size, local_addr(src_addr, noc.get_noc_id()));
}

template <uint32_t transfer_size>
FORCE_INLINE void read_with_state(Noc noc, uint32_t dst_addr, uint32_t src_addr) {
    UnicastEndpoint ep;
    noc.async_read_with_state<NocOptions::DEFAULT, transfer_size>(
        ep, CoreLocalMem<uint32_t>(dst_addr), transfer_size, local_addr(src_addr, noc.get_noc_id()), {});
}

template <typename Dst>
FORCE_INLINE void read_with_state(
    Noc noc, const Dst& dst, uint32_t src_addr, const typename noc_traits_t<Dst>::dst_args_type& dst_args) {
    UnicastEndpoint ep;
    // size_bytes = 0 is deliberate: max_page_size = 1 selects async_read_with_state's
    // single-packet branch, which takes the size from the state programmed by
    // set_read_state<transfer_size>() and ignores this argument.
    noc.async_read_with_state<NocOptions::DEFAULT, 1>(ep, dst, 0, local_addr(src_addr, noc.get_noc_id()), dst_args);
}

template <typename Dst>
FORCE_INLINE void read_with_state(Noc noc, const Dst& dst, uint32_t src_addr) {
    UnicastEndpoint ep;
    // size_bytes = 0 is deliberate — see the dst_args overload above.
    noc.async_read_with_state<NocOptions::DEFAULT, 1>(ep, dst, 0, local_addr(src_addr, noc.get_noc_id()), {});
}

// ─── set_read_trid / async_read_barrier_with_trid ───────────────────────────
//
// Example — a two-deep read pipeline that drains batch k while batch k+1 is in flight:
//
//   for (uint32_t issued = 0; issued < n_batches; ++issued) {
//       if (issued >= N_TRIDS) {
//           dataflow_kernel_lib::async_read_barrier_with_trid(noc, trid_for(issued - N_TRIDS));
//       }
//       dataflow_kernel_lib::set_read_trid(noc, trid_for(issued));
//       ... issue this batch's reads ...
//   }
//   ... drain the remaining trids ...
//   dataflow_kernel_lib::set_read_trid(noc, 0);  // restore untagged
//
FORCE_INLINE void set_read_trid(Noc noc, uint32_t trid) { noc_async_read_set_trid(trid, noc.get_noc_id()); }

FORCE_INLINE void async_read_barrier_with_trid(Noc noc, uint32_t trid) {
    noc.template async_read_barrier<NocOptions::TXN_ID>({.trid = trid});
}

}  // namespace dataflow_kernel_lib
