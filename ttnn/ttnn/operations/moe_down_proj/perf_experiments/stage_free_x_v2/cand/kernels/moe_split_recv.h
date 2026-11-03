// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Split receive for a ttnn.Mcast(handshake=True, Flag data-ready) channel: ack(round) and wait() as two calls, on the
// same wire as ReceiverPipe::receive(round) (which fuses them). A core may do other NoC work between the ack and the
// wait (the x forwarder forwards the previous step there). One round in flight, so the single data-ready cell holds.
// Bypasses dataflow_kernel_lib ReceiverPipe::receive(): the helper has no ack-only / wait-only entry points.

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

namespace moe_split_recv {

namespace dkl = dataflow_kernel_lib;
namespace wire = dataflow_kernel_lib::mcast_wire;

template <bool PRESENT, uint32_t CT, uint32_t RT>
struct Receiver {
    explicit Receiver(const Noc&) {}
    void ack(uint32_t) {}
    void wait() {}
    template <typename Work>
    void wait_doing(Work&&) {}
};

template <uint32_t CT, uint32_t RT>
struct Receiver<true, CT, RT> {
    using Args = dkl::McastArgs<CT, RT>;
    using Runtime = dkl::detail::PositionalMcastRuntime<RT>;
    static constexpr uint32_t DR_ID = dkl::detail::positional_mcast_semaphore<CT, wire::DATA_READY>();
    static constexpr uint32_t CR_ID = dkl::detail::positional_mcast_semaphore<CT, wire::CONSUMER_READY>();
    static_assert(Args::transfer_mode == dkl::TransferMode::Multicast, "hardware multicast only");
    static_assert(Args::pre_handshake && Args::signal == dkl::DataReadySignal::Flag, "handshake + Flag only");
    using Sem = Semaphore<ProgrammableCoreType::TENSIX>;
    using Dr = dkl::detail::McastSemaphoreToken<dkl::McastSemaphoreBinding{DR_ID}>;
    using Cr = dkl::detail::McastSemaphoreToken<dkl::McastSemaphoreBinding{CR_ID}>;

    explicit Receiver(const Noc& noc) :
        noc_(noc), dr_(dkl::detail::make_mcast_semaphore<Dr{}>()), cr_(dkl::detail::make_mcast_semaphore<Cr{}>()) {
        if constexpr (Args::receiver_available) {
            if (args_.can_receive()) {
                coords_.emplace(Runtime::coordinates(Args::runtime_layout.sender_coordinates));
            }
        }
    }
    void ack(uint32_t round) {
        if constexpr (Args::receiver_available) {
            const uint32_t idx = round % Args::num_senders;
            const auto& c = *coords_;
            cr_.up(
                noc_,
                c[wire::SENDER_COORD_WORDS * idx + wire::SENDER_X],
                c[wire::SENDER_COORD_WORDS * idx + wire::SENDER_Y],
                1);
        }
    }
    void wait() {
        dr_.wait(VALID);
        dr_.set(INVALID);
    }
    // wait() with `work` run between polls; work() returns false once it has nothing left to do.
    template <typename Work>
    void wait_doing(Work&& work) {
        while (dr_.value() != VALID) {
            if (!work()) {
                dr_.wait(VALID);
                break;
            }
        }
        dr_.set(INVALID);
    }

private:
    static constexpr Args args_{};
    Noc noc_;
    Sem dr_;
    Sem cr_;
    std::optional<typename Args::SenderCoordinates> coords_;
};

template <uint32_t CT, uint32_t RT>
using ReceiverFor = Receiver<get_compile_time_arg_val(CT) != wire::ABSENT, CT, RT>;

}  // namespace moe_split_recv
