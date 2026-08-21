// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// mcast_pipe_spec — the Metal 2.0 (ProgramSpec) face of mcast_pipe.
// =============================================================================
//
// Separate from mcast_pipe.hpp on purpose: this reads the vararg block via get_vararg(), which
// only exists in kernels built from a ProgramSpec. Its arguments are not type-dependent, so a
// template mentioning it is looked up at DEFINITION time — putting this in the shared header
// breaks every descriptor-path kernel that includes it. Spec kernels opt in by including this.
//
// The pipes themselves are unchanged: they take semaphore ids as template params and coords as
// constructor args, and never touch kernel args. Only where the wire lives differs from McastArgs:
//
//   CT: named args, not a positional block. Under a ProgramSpec, compile_time_args is a
//       Table<string, uint32_t> whose order is meaningless and whose values are emitted directly
//       as constants (there is no CT index to offset into), so each field arrives as its own named
//       arg and is passed in as a template param.
//   semaphore ids: from the kernel's SemaphoreBindings (`sem::<name>`), not from the host — the
//       host no longer picks ids at all.
//   RT: the vararg block, which IS ordered and 0-based independently of the named args, so the
//       RT_BASE chaining McastArgs does works unchanged.
// =============================================================================

#pragma once

#include "experimental/kernel_args.h"
#include "mcast_pipe.hpp"

namespace dataflow_kernel_lib {

template <
    uint32_t DATA_READY_SEM_ID,
    uint32_t CONSUMER_READY_SEM_ID,
    uint32_t ACTIVE,
    uint32_t NUM_ACTIVE,
    uint32_t FLAGS,
    uint32_t RT_BASE,
    uint32_t SPAN = 0>
struct McastArgsSpec {
    static constexpr uint32_t active = ACTIVE;
    static constexpr uint32_t data_ready = DATA_READY_SEM_ID;
    static constexpr uint32_t consumer_ready = CONSUMER_READY_SEM_ID;
    static constexpr uint32_t num_active = NUM_ACTIVE;
    static constexpr uint32_t flags = FLAGS;

    static constexpr bool pre_handshake = (flags & 0x1u) != 0u;
    static constexpr DataReadySignal signal =
        ((flags >> 1) & 0x1u) != 0u ? DataReadySignal::Counter : DataReadySignal::Flag;
    static constexpr bool rotating = SPAN > 0;
    static constexpr uint32_t num_senders = SPAN == 0 ? 1u : SPAN;

    static constexpr uint32_t num_runtime_varargs() { return SPAN == 0 ? 4u : (4u + 2u * SPAN); }
    static constexpr uint32_t next_runtime_varargs_offset() { return RT_BASE + num_runtime_varargs(); }

    template <uint8_t NOC_ID = noc_index>
    SenderPipe<NOC_ID, data_ready, pre_handshake, consumer_ready, signal, rotating> sender(const Noc& noc) const {
        return SenderPipe<NOC_ID, data_ready, pre_handshake, consumer_ready, signal, rotating>(
            noc, rect<NOC_ID>(), num_active);
    }

    // ReceiverPipe holds `sender_coords` as a NON-OWNING pointer and dereferences it in receive(),
    // so the storage must outlive the pipe: a local array built here dies with this frame and
    // receive() would then ack a garbage core, deadlocking the sender's pre-handshake wait. Point
    // at the L1 vararg block itself, which lives for the whole kernel -- the same thing McastArgs
    // does on the descriptor path with get_arg_addr(). The coord pairs already sit contiguously:
    // [sender_x, sender_y] at RT_BASE for a fixed sender, and one pair per round from RT_BASE + 4
    // in rotating mode (where RT_BASE..RT_BASE+3 is the rect).
    ReceiverPipe<data_ready, pre_handshake, consumer_ready, signal, num_senders> receiver(const Noc& noc) const {
        return ReceiverPipe<data_ready, pre_handshake, consumer_ready, signal, num_senders>(
            noc, get_vararg_addr(RT_BASE + (SPAN == 0 ? 0u : 4u)));
    }

    template <uint8_t NOC_ID = noc_index>
    McastRect<NOC_ID> rect() const {
        return McastRect<NOC_ID>(
            get_vararg(RT_BASE + 0), get_vararg(RT_BASE + 1), get_vararg(RT_BASE + 2), get_vararg(RT_BASE + 3));
    }
    uint32_t sender_x() const { return get_vararg(RT_BASE + 0); }
    uint32_t sender_y() const { return get_vararg(RT_BASE + 1); }
    uint32_t sender_x(uint32_t round) const { return get_vararg(RT_BASE + 4 + 2 * round + 0); }
    uint32_t sender_y(uint32_t round) const { return get_vararg(RT_BASE + 4 + 2 * round + 1); }
};

}  // namespace dataflow_kernel_lib

// Build the decoder for the family attached under `prefix`. Kernels compile as C++17 (no string
// non-type template params), so the prefix is pasted by the preprocessor rather than passed as a
// template argument.
//
// Every name it reads is emitted by McastFamily.attach() on the host: the two semaphore bindings and
// the four named compile-time args, including <prefix>_rt_base — which is why the caller never chains
// a vararg offset the way the descriptor path chains RT_BASE.
//
//   constexpr auto mc = MCAST_ARGS(row);
//   auto sender = mc.sender(noc);
#define MCAST_ARGS(prefix)                  \
    ::dataflow_kernel_lib::McastArgsSpec<   \
        sem::prefix##_data_ready,           \
        sem::prefix##_consumer_ready,       \
        get_arg(args::prefix##_active),     \
        get_arg(args::prefix##_num_active), \
        get_arg(args::prefix##_flags),      \
        get_arg(args::prefix##_rt_base)> {}

// Rotating variant: `span` is the round count (== the broadcast span), which sizes the RT block.
#define MCAST_ARGS_ROTATING(prefix, span)   \
    ::dataflow_kernel_lib::McastArgsSpec<   \
        sem::prefix##_data_ready,           \
        sem::prefix##_consumer_ready,       \
        get_arg(args::prefix##_active),     \
        get_arg(args::prefix##_num_active), \
        get_arg(args::prefix##_flags),      \
        get_arg(args::prefix##_rt_base),    \
        (span)> {}
