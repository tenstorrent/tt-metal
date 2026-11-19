// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "flags.hpp"
#include "relay_port.hpp"

void kernel_main() {
    Noc noc;
    constexpr uint32_t tile_bytes = CT_ARG("tile_bytes_bf16");
    constexpr uint32_t b_tiles = CT_ARG("b_tiles_per_core");
    constexpr uint32_t segment_bytes = b_tiles * tile_bytes;
    constexpr uint32_t chips = CT_ARG("chips");
    constexpr uint32_t lag = CT_ARG("owner_lag");
    constexpr uint32_t reduce_lag = CT_ARG("reduce_lag");
    constexpr uint32_t rows = CT_ARG("rows");
    constexpr uint32_t depth = CT_ARG("ring_depth");
    const uint32_t chip = RT_ARG("chip");
    const uint32_t core = RT_ARG("index");
    const uint32_t self_x = RT_ARG("self_x");
    const uint32_t self_y = RT_ARG("self_y");
    const uint32_t first_column = core * b_tiles;
    const auto y = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_y")>(), RT_ARG("y_addr"), tile_bytes);
    constexpr uint32_t k_groups = CT_ARG("k_groups");
    constexpr uint32_t kred_depth = CT_ARG("kred_depth");
    const uint32_t k_group = core % k_groups;
    const uint32_t n_group = core / k_groups;
    CircularBuffer send(CT_ARG("cb_send"));
    CircularBuffer slabs(CT_ARG("cb_kout"));
    FlagBoard flags{arena_base(CT_ARG("cb_flags")), 8, 0};
    const uint32_t kred_landing[k_groups - 1] = {
        arena_base(CT_ARG("cb_kred")), arena_base(CT_ARG("cb_kred") + 1), arena_base(CT_ARG("cb_kred") + 2)};
    CircularBuffer y_segment(CT_ARG("cb_y"));

    const uint32_t relay_slots = arena_base(CT_ARG("cb_a")) + (core >> 1) * 2 * CT_ARG("relay_slot_bytes");
    const uint32_t rs_landing[RING_CHIPS - 1] = {
        arena_base(CT_ARG("cb_rs")), arena_base(CT_ARG("cb_rs") + 1), arena_base(CT_ARG("cb_rs") + 2)};
    const uint32_t ag_landing = arena_base(CT_ARG("cb_ag"));
    auto make_port = [&](uint32_t direction) {
        return RelayPort{
            CT_ARG("relay_grid_x_start") + direction * 2 + (core & 1),
            CT_ARG("relay_grid_y_start"),
            relay_slots,
            arena_base(CT_ARG("cb_message")),
            Semaphore<>(CT_ARG("sem_relay_free") + direction),
            0};
    };
    RelayPort ports[2] = {make_port(0), make_port(1)};
    auto semaphore_address = [](uint32_t base, uint32_t index) { return get_semaphore(base + index); };

    auto write_segment = [&](uint32_t m, uint32_t source) {
        for (uint32_t t = 0; t < b_tiles; t++) {
            noc.async_write(
                CoreLocalMem<uint32_t>(source + t * tile_bytes),
                y,
                tile_bytes,
                {},
                {.page_id = m * CT_ARG("h_tiles") + first_column + t});
        }
    };
    auto send_to = [&](uint32_t peer,
                       uint32_t q,
                       MessageKind kind,
                       uint32_t payload,
                       uint32_t payload_bytes,
                       uint32_t dst_address,
                       uint32_t semaphore) {
        const RingRoute route = ring_route(chip, peer, core);
        ports[route.relay].push(noc, kind, route.hops, payload, payload_bytes, self_x, self_y, dst_address, semaphore);
    };

    for (uint32_t step = 0; step < rows + reduce_lag + lag; step++) {
        if (step < rows) {
            slabs.wait_front((k_groups - 1) * b_tiles);
            for (uint32_t outgoing = 0; outgoing < k_groups - 1; outgoing++) {
                const uint32_t group = outgoing < k_group ? outgoing : outgoing + 1;
                const uint32_t peer_x = CT_ARG("b_grid_x_start") + group;
                const uint32_t peer_y = CT_ARG("b_grid_y_start") + n_group;
                flags.wait_min(FlagBoard::credit + peer_index(group, k_group), step + 1);
                noc.async_write(
                    CoreLocalMem<uint32_t>(slabs.get_read_ptr() + outgoing * segment_bytes),
                    UnicastEndpoint{},
                    segment_bytes,
                    {},
                    {.noc_x = peer_x,
                     .noc_y = peer_y,
                     .addr = kred_landing[peer_index(k_group, group)] + (step % kred_depth) * segment_bytes});
                flags.post(noc, peer_x, peer_y, FlagBoard::ready + peer_index(k_group, group), step + 1);
            }
            noc.async_writes_flushed();
            slabs.pop_front((k_groups - 1) * b_tiles);
        }
        if (chips > 1 && step >= reduce_lag && step - reduce_lag < rows && (step - reduce_lag) % chips != chip) {
            const uint32_t m = step - reduce_lag;
            const uint32_t owner = m % chips;
            const uint32_t q = m / chips;
            const uint32_t me = peer_index(chip, owner);
            send.wait_front(b_tiles);
            if (q + 1 > depth) {
                Semaphore<>(CT_ARG("sem_credit_rs") + peer_index(owner, chip)).wait_min(q + 1 - depth);
            }
            send_to(
                owner,
                q,
                MessageKind::Data,
                send.get_read_ptr(),
                segment_bytes,
                rs_landing[me] + (q % depth) * segment_bytes,
                semaphore_address(CT_ARG("sem_rs_ready"), me));
            send.pop_front(b_tiles);
        }
        if (step >= reduce_lag + lag) {
            const uint32_t m = step - reduce_lag - lag;
            const uint32_t owner = m % chips;
            const uint32_t q = m / chips;
            if (owner == chip) {
                y_segment.wait_front(b_tiles);
                write_segment(m, y_segment.get_read_ptr());
                for (uint32_t offset = 1; chips > 1 && offset < chips; offset++) {
                    const uint32_t receiver = (chip + offset) % chips;
                    const uint32_t me = peer_index(chip, receiver);
                    if (q + 1 > depth) {
                        Semaphore<>(CT_ARG("sem_credit_ag") + peer_index(receiver, chip)).wait_min(q + 1 - depth);
                    }
                    send_to(
                        receiver,
                        q,
                        MessageKind::Data,
                        y_segment.get_read_ptr(),
                        segment_bytes,
                        ag_landing + (me * depth + q % depth) * segment_bytes,
                        semaphore_address(CT_ARG("sem_ag_ready"), me));
                }
                noc.async_write_barrier();
                y_segment.pop_front(b_tiles);
                for (uint32_t offset = 1; chips > 1 && offset < chips; offset++) {
                    const uint32_t sender = (chip + offset) % chips;
                    send_to(
                        sender,
                        q,
                        MessageKind::Increment,
                        0,
                        0,
                        0,
                        semaphore_address(CT_ARG("sem_credit_rs"), peer_index(chip, sender)));
                }
            } else {
                const uint32_t me = peer_index(owner, chip);
                Semaphore<>(CT_ARG("sem_ag_ready") + me).wait_min(q + 1);
                write_segment(m, ag_landing + (me * depth + q % depth) * segment_bytes);
                noc.async_write_barrier();
                send_to(
                    owner,
                    q,
                    MessageKind::Increment,
                    0,
                    0,
                    0,
                    semaphore_address(CT_ARG("sem_credit_ag"), peer_index(chip, owner)));
            }
        }
    }
    noc.async_write_barrier();
    for (uint32_t direction = 0; chips > 1 && direction < 2; direction++) {
        ports[direction].push(noc, MessageKind::End, 0, 0, 0, 0, 0, 0, 0);
    }
}
