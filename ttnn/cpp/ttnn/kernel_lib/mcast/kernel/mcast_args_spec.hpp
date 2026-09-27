// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/mcast_spec_common.hpp"

namespace dataflow_kernel_lib::detail {

// Native get_vararg accounts for the final named-RTA section. The view keeps only an
// offset, is copied into the receiver pipe, and never borrows the decoder's storage.
template <uint32_t BASE>
struct SpecMcastRuntime {
    uint32_t offset;
    static uint32_t read(uint32_t index) { return get_vararg(BASE + index); }
    static SpecMcastRuntime coordinates(uint32_t index) { return {index}; }
    uint32_t operator[](uint32_t index) const { return read(offset + index); }
};

}  // namespace dataflow_kernel_lib::detail

#define TT_MCAST_SPEC_READ_METADATA(prefix, field)                                           \
    .field = static_cast<decltype(dataflow_kernel_lib::mcast_wire::FamilyMetadata{}.field)>( \
        get_arg(args::TT_MCAST_SPEC_NAME(prefix, field))),

// The three *_type compiler definitions name native sem::<accessor>_t aliases, or
// std::nullptr_t for unused roles. Absent channels instantiate no resource operations.
#define MCAST_ARGS(prefix)                                                                                            \
    dataflow_kernel_lib::detail::McastArgsImpl<                                                                       \
        (get_arg(args::TT_MCAST_SPEC_NAME(prefix, tag)) == dataflow_kernel_lib::mcast_wire::FAMILY),                  \
        dataflow_kernel_lib::mcast_wire::FamilyMetadata{TT_MCAST_SPEC_METADATA(TT_MCAST_SPEC_READ_METADATA, prefix)}, \
        dataflow_kernel_lib::detail::SpecMcastRuntime<get_arg(args::TT_MCAST_SPEC_NAME(prefix, rt_base))>,            \
        TT_MCAST_SPEC_NAME(prefix, data_ready_type){},                                                                \
        TT_MCAST_SPEC_NAME(prefix, consumer_ready_type){},                                                            \
        TT_MCAST_SPEC_NAME(prefix, signal_source_type){}> {}
