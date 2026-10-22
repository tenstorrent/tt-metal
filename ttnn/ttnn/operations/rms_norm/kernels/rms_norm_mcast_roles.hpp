// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Multicast role pipes for a kernel source that the host may attach to TWO kernel descriptors on different NoCs
// (Perf 2, dram_noc_balance: the NOC_SWAP_COLUMNS westmost Tensix columns of an interleaved input run this source with
// the reader on NoC1 / writer on NoC0; every other core runs it on the default NoCs). A multicast channel attached to
// both descriptors compiles `sender()` / `receiver()` in a kernel whose cores may hold no such role, and
// `McastArgs::sender()` static_asserts `sender_available` and `sender_noc == noc_index` even inside a runtime branch
// that those cores never take. These wrappers hand such a kernel an inert stand-in instead. The host never swaps a
// multicast sender (noc_swap_cores), so the stand-in is never executed; on the default-NoC kernel the wrappers expand
// to exactly `mc.sender(noc)` / `mc.receiver(noc)`.
// Include after api/dataflow/noc.h and kernel_lib/mcast/kernel/mcast_args.hpp.

#pragma once

#include <stdint.h>

namespace rms_norm_mcast {

struct InertSender {
    template <auto... Ts, typename... A>
    FORCE_INLINE void send(A...) {}
    template <auto... Ts, typename... A>
    FORCE_INLINE void send_signal(A...) {}
};
struct InertReceiver {
    template <auto... Ts, typename... A>
    FORCE_INLINE void receive(A...) {}
};

template <class MC>
FORCE_INLINE auto sender(const MC& mc, const Noc& noc) {
    if constexpr (MC::sender_available && MC::sender_noc == noc_index) {
        return mc.sender(noc);
    } else {
        return InertSender{};
    }
}

template <class MC>
FORCE_INLINE auto receiver(const MC& mc, const Noc& noc) {
    if constexpr (MC::receiver_available) {
        return mc.receiver(noc);
    } else {
        return InertReceiver{};
    }
}

}  // namespace rms_norm_mcast
