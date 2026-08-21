// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file local_copy_helpers_dataflow.hpp
 * @brief Local L1 -> L1 copy helpers: self-aimed NoC reads, stateful single-packet reads,
 *        and per-batch transaction-id (trid) tagging.
 *
 * WHY A READ AND NOT A WRITE
 * --------------------------
 * A self-aimed READ is the only way to copy L1 -> L1 with a typed buffer (DataflowBuffer /
 * CircularBuffer) on one side. Noc::async_read resolves its DESTINATION as
 * Noc::AddressType::LOCAL_L1 (api/dataflow/noc.h, `get_dst_ptr<AddressType::LOCAL_L1>` in
 * async_read), which is exactly what noc_traits_t<DataflowBuffer>::dst_addr and
 * noc_traits_t<CircularBuffer>::dst_addr static_assert on
 * (api/dataflow/dataflow_buffer.h `dst_addr`, api/dataflow/circular_buffer.h `dst_addr`).
 * Noc::async_write resolves its destination as AddressType::NOC instead, so a
 * DataflowBuffer/CircularBuffer passed as a unicast write destination fails that static_assert.
 * Hence: the buffer stays the typed DESTINATION of a read, and only the SOURCE is a raw address
 * turned into a self-aimed UnicastEndpoint by local_addr().
 *
 * WHAT THE THREE PIECES ARE FOR
 * -----------------------------
 *   local_addr()      — build the self-aimed UnicastEndpoint src_args for any Noc read call.
 *   set_read_state() +
 *   read_with_state() — amortise the NoC config-register setup across a copy loop: program the
 *                       source and size ONCE, then issue N reads that only write the
 *                       destination register.
 *   set_read_trid() /
 *   async_read_barrier_with_trid() — tag a batch of reads and wait on just that batch, so a
 *                       pipelined loop can drain batch k while batch k+1 is still in flight.
 */

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

namespace dataflow_kernel_lib {

/**
 * @brief Build UnicastEndpoint source arguments aimed at this core's own L1.
 *
 * WARNING — NOC 0 AND NOC 1 HAVE DIFFERENT COORDINATE SPACES. my_x/my_y must be indexed by the
 * id of the NoC that will actually issue the transfer, never by a hardcoded [0]. The default
 * `noc_id = noc_index` is the RISC-V's own default NoC and is correct for a bare `Noc noc;`.
 * When the transfer runs on an explicitly-constructed Noc, pass `noc.get_noc_id()`.
 *
 * @param addr Local L1 byte address to read from
 * @param noc_id NoC whose coordinate space `addr` is expressed in (default: this core's noc_index)
 * @return src_args for UnicastEndpoint, pointing at (my_x[noc_id], my_y[noc_id], addr)
 */
FORCE_INLINE noc_traits_t<UnicastEndpoint>::src_args_type local_addr(uint32_t addr, uint8_t noc_id = noc_index);

/**
 * @brief Program the stateful read registers for a single-packet self-aimed read loop.
 *
 * Call once, then call read_with_state() repeatedly. `transfer_size` is a template parameter so
 * that this call and every matching read_with_state() agree on single-packet mode: the size is
 * BAKED INTO THE STATE here and is therefore NOT re-sent per read.
 *
 * SINGLE-PACKET ONLY: the static_assert bounds `transfer_size` by NOC_MAX_BURST_SIZE. The
 * stateful path only programs one packet's worth of state, so multi-packet transfers must go
 * through plain Noc::async_read instead.
 *
 * @tparam transfer_size Bytes per read; must be <= NOC_MAX_BURST_SIZE
 * @param noc NoC that will issue the reads (also selects the coordinate space for `src_addr`)
 * @param src_addr Local L1 byte address to read from
 */
template <uint32_t transfer_size>
FORCE_INLINE void set_read_state(Noc noc, uint32_t src_addr);

/**
 * @brief Issue one self-aimed read into a raw local L1 address, using previously set state.
 *
 * `transfer_size` must match the value passed to set_read_state(). The default of 1 selects the
 * single-packet branch (`max_page_size <= NOC_MAX_BURST_SIZE`) without claiming a specific size,
 * which is what callers that only ever set one state want.
 *
 * The template argument doubles as the size_bytes argument passed through to
 * Noc::async_read_with_state. Any value <= NOC_MAX_BURST_SIZE selects the single-packet branch,
 * which reads the size out of the state and ignores size_bytes — which is why the default of 1
 * is safe for a state programmed with a larger transfer_size.
 *
 * @tparam transfer_size Must match set_read_state()'s transfer_size (default: 1)
 * @param noc NoC that was programmed by set_read_state()
 * @param dst_addr Local L1 byte address to write into
 * @param src_addr Local L1 byte address to read from (must be the address set_read_state() programmed)
 */
template <uint32_t transfer_size = 1>
FORCE_INLINE void read_with_state(Noc noc, uint32_t dst_addr, uint32_t src_addr);

/**
 * @brief Issue one self-aimed read into a typed local L1 destination, with destination arguments.
 *
 * `Dst` is any type whose noc_traits_t resolves a LOCAL_L1 destination address —
 * DataflowBuffer, CircularBuffer, CoreLocalMem. This is the overload that makes an L1 -> L1 copy
 * expressible against a buffer object at all (see the file-level "WHY A READ AND NOT A WRITE").
 *
 * NOTE — size_bytes IS DELIBERATELY 0. The transfer size was baked into the NoC state by
 * set_read_state<transfer_size>(). This overload pins Noc::async_read_with_state's max_page_size
 * to 1, which selects its single-packet branch (noc_async_read_one_packet_with_state) — and that
 * branch IGNORES the size_bytes argument entirely, taking the size from the state. Passing 0
 * documents "the state owns the size"; do not "fix" it to a real byte count.
 *
 * @tparam Dst Typed local-L1 destination (deduced)
 * @param noc NoC that was programmed by set_read_state()
 * @param dst Destination buffer
 * @param src_addr Local L1 byte address to read from (must be the address set_read_state() programmed)
 * @param dst_args Destination arguments, e.g. `{.offset_bytes = ...}`
 */
template <typename Dst>
FORCE_INLINE void read_with_state(
    Noc noc, const Dst& dst, uint32_t src_addr, const typename noc_traits_t<Dst>::dst_args_type& dst_args);

/**
 * @brief Issue one self-aimed read into a typed local L1 destination, at its base address.
 *
 * Same contract as the dst_args overload (including the deliberate size_bytes = 0 and the
 * single-packet branch that ignores it); the destination arguments default-construct, i.e.
 * offset_bytes = 0.
 *
 * @tparam Dst Typed local-L1 destination (deduced)
 * @param noc NoC that was programmed by set_read_state()
 * @param dst Destination buffer
 * @param src_addr Local L1 byte address to read from (must be the address set_read_state() programmed)
 */
template <typename Dst>
FORCE_INLINE void read_with_state(Noc noc, const Dst& dst, uint32_t src_addr);

/**
 * @brief Set the active transaction id (NOC_PACKET_TAG) for subsequent async_read* calls on this
 *        Noc's read cmd_buf.
 *
 * The trid persists across set_read_state() / read_with_state() — those write different cmd_buf
 * registers — so one set_read_trid() tags a whole batch. Pair with
 * async_read_barrier_with_trid() to wait on just that batch. Pass trid = 0 to clear (untagged
 * reads get no per-trid accounting).
 *
 * @param noc NoC whose read cmd_buf is tagged
 * @param trid Transaction id, or 0 to clear
 */
FORCE_INLINE void set_read_trid(Noc noc, uint32_t trid);

/**
 * @brief Block until the reads tagged `trid` on this NoC are flushed.
 *
 * Other in-flight reads carrying different trids continue independently — that is the point:
 * a pipelined loop drains batch k here while batch k+1 is still outstanding.
 *
 * @param noc NoC to wait on
 * @param trid Transaction id to wait for
 */
FORCE_INLINE void async_read_barrier_with_trid(Noc noc, uint32_t trid);

}  // namespace dataflow_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.inl"
