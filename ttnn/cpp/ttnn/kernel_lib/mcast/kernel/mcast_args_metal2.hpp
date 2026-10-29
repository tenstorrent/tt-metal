// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/mcast_common_metal2.hpp"

namespace dataflow_kernel_lib::detail {

template <uint32_t BASE>
struct Metal2McastCompileTime {
    constexpr uint32_t operator[](uint32_t index) const { return get_compile_time_vararg(BASE + index); }
};

// Native get_vararg accounts for the final named-RTA section. The view keeps only an
// offset, is copied into the receiver pipe, and never borrows the decoder's storage.
template <uint32_t BASE>
struct Metal2McastRuntime {
    uint32_t offset;
    static uint32_t read(uint32_t index) { return get_vararg(BASE + index); }
    static Metal2McastRuntime coordinates(uint32_t index) { return {index}; }
    uint32_t operator[](uint32_t index) const { return read(offset + index); }
};

}  // namespace dataflow_kernel_lib::detail

// Metal 2.0 usage. The prefix must match the host-side Mcast::attach() prefix:
//
//   constexpr auto channel = MCAST_ARGS(channel);
//   Noc noc;
//
//   auto sender = channel.sender(noc);
//   sender.send(src_l1, dst_l1, size_bytes);
//
//   auto receiver = channel.receiver(noc);
//   receiver.receive(round);
//
// The three *_type compiler definitions name native sem::<accessor>_t aliases, or
// std::nullptr_t for unused roles. Absent channels instantiate no resource operations.
#define MCAST_ARGS(prefix)                                                                                             \
    dataflow_kernel_lib::detail::McastArgsImpl<                                                                        \
        (get_compile_time_vararg(get_arg(args::TT_MCAST_METAL2_NAME(prefix, ct_base))) !=                              \
         dataflow_kernel_lib::mcast_wire::ABSENT),                                                                     \
        dataflow_kernel_lib::detail::mcast_metadata<                                                                   \
            dataflow_kernel_lib::detail::Metal2McastCompileTime<get_arg(args::TT_MCAST_METAL2_NAME(prefix, ct_base))>, \
            false>(),                                                                                                  \
        dataflow_kernel_lib::detail::Metal2McastRuntime<get_arg(args::TT_MCAST_METAL2_NAME(prefix, rt_base))>,         \
        TT_MCAST_METAL2_NAME(prefix, data_ready_type),                                                                 \
        TT_MCAST_METAL2_NAME(prefix, consumer_ready_type),                                                             \
        TT_MCAST_METAL2_NAME(prefix, signal_source_type)> {}
