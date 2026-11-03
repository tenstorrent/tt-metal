// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Early-ack multicast channel (perf experiment, op-local). Same wire as a host ttnn.Mcast(handshake=True,
// Flag data-ready) channel, plus one extra data-ready cell (DR1_ID) so two consecutive rounds can be in
// flight: round r signals cell r % 2. The receiver API is split into ack(round) and wait(round), so a
// receiver acks round r + 1 (to its rotating sender) as soon as its slot is free, before round r's data
// lands. Bypasses dataflow_kernel_lib ReceiverPipe::receive(), which fuses ack + wait on one data-ready cell.
// Lookahead is one round: a sender's consumer-ready counter and a data-ready cell are reused every
// num_senders and 2 rounds, so this needs num_senders >= 2 (asserted).

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

namespace moe_early_ack {

namespace dkl = dataflow_kernel_lib;
namespace wire = dataflow_kernel_lib::mcast_wire;

template <bool PRESENT, uint32_t CT, uint32_t RT, uint32_t DR1_ID>
struct Channel;

template <uint32_t CT, uint32_t RT, uint32_t DR1_ID>
struct Channel<false, CT, RT, DR1_ID> {
    explicit Channel(const Noc&) {}
    bool can_send() const { return false; }
    bool can_receive() const { return false; }
    bool should_send(uint32_t) const { return false; }
    void ack(uint32_t) {}
    bool ready(uint32_t) const { return true; }
    bool acks_ready() const { return true; }
    void consume(uint32_t) {}
    void wait(uint32_t) {}
    void send(uint32_t, uint32_t, uint32_t, uint32_t) {}
    void issue(uint32_t, uint32_t, uint32_t, uint32_t) {}
    void wait_acks() {}
    void issue_data(uint32_t, uint32_t, uint32_t, uint32_t) {}
    void retire(uint32_t) {}
};

template <uint32_t CT, uint32_t RT, uint32_t DR1_ID>
struct Channel<true, CT, RT, DR1_ID> {
    using Args = dkl::McastArgs<CT, RT>;
    using Runtime = dkl::detail::PositionalMcastRuntime<RT>;
    static constexpr auto META = dkl::detail::mcast_metadata<dkl::detail::PositionalMcastCompileTime<CT>>();
    static constexpr uint32_t DR0_ID = dkl::detail::positional_mcast_semaphore<CT, wire::DATA_READY>();
    static constexpr uint32_t CR_ID = dkl::detail::positional_mcast_semaphore<CT, wire::CONSUMER_READY>();
    using Dr0 = dkl::detail::McastSemaphoreToken<dkl::McastSemaphoreBinding{DR0_ID}>;
    using Dr1 = dkl::detail::McastSemaphoreToken<dkl::McastSemaphoreBinding{DR1_ID}>;
    using Cr = dkl::detail::McastSemaphoreToken<dkl::McastSemaphoreBinding{CR_ID}>;
    static constexpr uint32_t CAP = Args::rectangle_capacity ? Args::rectangle_capacity : 1;
    static_assert(Args::transfer_mode == dkl::TransferMode::Multicast, "hardware multicast only");
    static_assert(Args::pre_handshake && Args::signal == dkl::DataReadySignal::Flag, "handshake + Flag only");
    static_assert(Args::num_senders >= 2, "one-round lookahead reuses a sender's ack counter every num_senders rounds");

    template <typename Dr>
    using Sender = dkl::detail::SenderPipeImpl<
        noc_index,
        Dr,
        true,
        Cr,
        dkl::DataReadySignal::Flag,
        Args::rotating,
        Args::sender_mcast_mode,
        CAP>;
    using Coords = typename Args::SenderCoordinates;

    explicit Channel(const Noc& noc) :
        noc_(noc),
        dr0_(dkl::detail::make_mcast_semaphore<Dr0{}>()),
        dr1_(dkl::detail::make_mcast_semaphore<Dr1{}>()),
        cr_(dkl::detail::make_mcast_semaphore<Cr{}>()) {
        if constexpr (Args::receiver_available) {
            if (args_.can_receive()) {
                coords_.emplace(Runtime::coordinates(Args::runtime_layout.sender_coordinates));
            }
        }
        if constexpr (Args::sender_available) {
            if (args_.can_send()) {
                const auto prepared = sender_args();
                ack_count_ = prepared.ack_count;
                sargs_ = prepared;
                s0_.emplace(noc, prepared);
                s1_.emplace(noc, prepared);
            }
        }
    }

    bool can_send() const { return args_.can_send(); }
    bool can_receive() const { return args_.can_receive(); }
    bool should_send(uint32_t round) const { return args_.should_send(round); }

    // "My slot for `round` is free": remote atomic inc on that round's sender's consumer-ready counter.
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
    // Sender side: every receiver of this sender's next round has acked (send() would not block).
    bool acks_ready() const { return cr_.value() >= ack_count_; }
    bool ready(uint32_t round) const { return cell(round).value() == VALID; }
    void consume(uint32_t round) { cell(round).set(INVALID); }
    void wait(uint32_t round) {
        cell(round).wait(VALID);
        cell(round).set(INVALID);
    }

    void send(uint32_t round, uint32_t src, uint32_t dst, uint32_t size) {
        if constexpr (Args::sender_available) {
            if (round & 1) {
                s1_->send(src, dst, size);
            } else {
                s0_->send(src, dst, size);
            }
        }
    }

    // Split send (PAIR_SEND): issue() = handshake + payload + flag with no fence, so two channels' sends from one
    // source overlap; the caller then runs one noc.async_write_barrier() and retire()s both. Same wire as send():
    // linked payload multicast terminated by the unlinked flag multicast from this round's cell.
    void wait_acks() {
        if constexpr (Args::sender_available) {
            cr_.wait(ack_count_);
            cr_.set(0);
        }
    }
    void issue(uint32_t round, uint32_t src, uint32_t dst, uint32_t size) {
        wait_acks();
        issue_data(round, src, dst, size);
    }
    void issue_data(uint32_t round, uint32_t src, uint32_t dst, uint32_t size) {
        if constexpr (Args::sender_available) {
            Sem& flag = cell(round);
            for (uint32_t i = 0; i < sargs_.num_rectangles; ++i) {
                const auto& rect = sargs_.rectangles[i];
                const auto& r = rect.bounds;
                ASSERT(rect.sender_mcast_mode != dkl::SenderMcastMode::LocalCopy);
                const bool loopback =
                    rect.sender_mcast_mode == dkl::SenderMcastMode::MulticastIncludeSource && src != dst;
                const uint32_t dests = loopback ? rect.loopback_count : rect.remote_count;
                UnicastEndpoint src_ep;
                MulticastEndpoint dst_ep;
                const typename noc_traits_t<UnicastEndpoint>::src_args_type src_args{.addr = src};
                const typename noc_traits_t<MulticastEndpoint>::dst_args_mcast_type dst_args{
                    r.sx, r.sy, r.ex, r.ey, dst};
                flag.set(VALID);
                if (loopback) {
                    noc_.template async_write_multicast<NocOptions::MCAST_INCL_SRC>(
                        src_ep, dst_ep, size, dests, src_args, dst_args, true);
                    flag.template set_multicast<NocOptions::MCAST_INCL_SRC>(noc_, r.sx, r.sy, r.ex, r.ey, dests, false);
                } else {
                    noc_.template async_write_multicast<NocOptions::DEFAULT>(
                        src_ep, dst_ep, size, dests, src_args, dst_args, true);
                    flag.template set_multicast<NocOptions::DEFAULT>(noc_, r.sx, r.sy, r.ex, r.ey, dests, false);
                }
            }
        }
    }
    void retire(uint32_t round) { cell(round).set(INVALID); }

private:
    using Sem = Semaphore<ProgrammableCoreType::TENSIX>;
    const Sem& cell(uint32_t round) const { return (round & 1) ? dr1_ : dr0_; }
    Sem& cell(uint32_t round) { return (round & 1) ? dr1_ : dr0_; }

    // Mirrors McastArgsImpl::sender_runtime_arguments() (private there).
    static dkl::SenderRuntimeArgumentsFor<CAP> sender_args() {
        constexpr auto layout = Args::runtime_layout;
        constexpr bool single = CAP == 1;
        dkl::SenderRuntimeArgumentsFor<CAP> prepared;
        prepared.num_rectangles = single ? 1u : Runtime::read(layout.rectangle_count);
        prepared.ack_count = Args::ack_count != dkl::ACK_EQUALS_FANOUT ? Args::ack_count : Runtime::read(layout.ack);
        for (uint32_t i = 0; i < prepared.num_rectangles; ++i) {
            const uint32_t base = layout.rectangles + i * layout.rectangle_stride;
            auto& rect = prepared.rectangles[i];
            if constexpr (layout.rectangle_bounds != wire::OMITTED) {
                rect.bounds = {
                    Runtime::read(base + layout.rectangle_bounds + wire::SX),
                    Runtime::read(base + layout.rectangle_bounds + wire::SY),
                    Runtime::read(base + layout.rectangle_bounds + wire::EX),
                    Runtime::read(base + layout.rectangle_bounds + wire::EY)};
            }
            rect.remote_count = single && META.mcast.remote_count_known ? Args::remote_count
                                                                        : Runtime::read(base + layout.rectangle_remote);
            rect.loopback_count = rect.remote_count + 1;
            rect.sender_mcast_mode =
                wire::concrete(Args::sender_mcast_mode)
                    ? Args::sender_mcast_mode
                    : static_cast<dkl::SenderMcastMode>(Runtime::read(base + layout.rectangle_mode));
        }
        return prepared;
    }

    static constexpr Args args_{};
    Noc noc_;
    Sem dr0_;
    Sem dr1_;
    Sem cr_;
    uint32_t ack_count_ = 0;
    dkl::SenderRuntimeArgumentsFor<CAP> sargs_{};
    std::optional<Coords> coords_;
    std::optional<Sender<Dr0>> s0_;
    std::optional<Sender<Dr1>> s1_;
};

template <uint32_t CT, uint32_t RT, uint32_t DR1_ID>
using ChannelFor = Channel<get_compile_time_arg_val(CT) != wire::ABSENT, CT, RT, DR1_ID>;

}  // namespace moe_early_ack
