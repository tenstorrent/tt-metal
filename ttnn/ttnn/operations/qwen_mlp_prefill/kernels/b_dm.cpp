// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "relay_port.hpp"

#include "weights.hpp"

void kernel_main() {
    Noc noc;
    constexpr uint32_t tile_bytes = CT_ARG("tile_bytes_bf16");
    constexpr uint32_t columns = CT_ARG("b_tiles_per_core");
    constexpr uint32_t segment_bytes = columns * tile_bytes;
    constexpr uint32_t chips = CT_ARG("chips");
    constexpr uint32_t rows = CT_ARG("rows");
    constexpr uint32_t depth = CT_ARG("landing_depth");
    constexpr uint32_t a_depth = CT_ARG("a_depth");
    constexpr uint32_t peers = chips - 1;
    constexpr bool owner = CT_ARG("owner") == 1;
    const uint32_t chip = RT_ARG("chip");
    const uint32_t first_column = RT_ARG("first_column");
    const uint32_t self_y = RT_ARG("self_y");
    const Rect a_collectors = RECT("a_collectors");
    CircularBuffer a(CT_ARG("cb_a"));
    CircularBuffer h_segment(CT_ARG("cb_h"));
    CircularBuffer y_segment(CT_ARG("cb_y"));
    CircularBuffer send(CT_ARG("cb_send"));
    Semaphore<> a_valid[a_depth] = {Semaphore<>(CT_ARG("sem_a_valid")), Semaphore<>(CT_ARG("sem_a_valid") + 1)};
    Semaphore<> land_ready[RING_CHIPS - 1] = {
        Semaphore<>(CT_ARG("sem_land_ready")),
        Semaphore<>(CT_ARG("sem_land_ready") + 1),
        Semaphore<>(CT_ARG("sem_land_ready") + 2)};
    Semaphore<> credit_y[RING_CHIPS - 1] = {
        Semaphore<>(CT_ARG("sem_credit_y")),
        Semaphore<>(CT_ARG("sem_credit_y") + 1),
        Semaphore<>(CT_ARG("sem_credit_y") + 2)};
    Semaphore<> credit_partial(CT_ARG("sem_credit_partial"));
    const uint32_t landing[RING_CHIPS - 1] = {
        arena_base(CT_ARG("cb_landing")), arena_base(CT_ARG("cb_landing") + 1), arena_base(CT_ARG("cb_landing") + 2)};
    const uint32_t group_x[RING_CHIPS] = {
        CT_ARG("b_group_x0"), CT_ARG("b_group_x1"), CT_ARG("b_group_x2"), CT_ARG("b_group_x3")};

    const auto h = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_h")>(), RT_ARG("h_addr"), tile_bytes);
    const auto y = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_y")>(), RT_ARG("y_addr"), tile_bytes);
    const auto w_down =
        TensorAccessor(TensorAccessorArgs<CT_ARG("ta_w_down")>(), RT_ARG("w_down_addr"), CT_ARG("tile_bytes_bfp4"));
    load_resident_weights(
        noc,
        w_down,
        CT_ARG("cb_wd"),
        CT_ARG("tile_bytes_bfp4"),
        CT_ARG("i_tiles"),
        CT_ARG("h_tiles"),
        0,
        first_column,
        columns);

    auto make_port = [&](uint32_t x, uint32_t source, uint32_t port) {
        return RelayPort{
            x,
            CT_ARG("relay_y"),
            arena_base(CT_ARG("cb_relay_slots")) + source * 2 * CT_ARG("relay_slot_bytes"),
            arena_base(CT_ARG("cb_message")),
            Semaphore<>(CT_ARG("sem_relay_free") + port),
            0};
    };
    RelayPort ports[2] = {
        make_port(RT_ARG("relay_x0"), RT_ARG("relay_source0"), 0),
        make_port(RT_ARG("relay_x1"), RT_ARG("relay_source1"), 1)};
    auto semaphore_address = [](uint32_t base, uint32_t index) { return get_semaphore(base + index); };
    auto write_segment = [&](uint32_t m, uint32_t source) {
        for (uint32_t t = 0; t < columns; t++) {
            noc.async_write(
                CoreLocalMem<uint32_t>(source + t * tile_bytes),
                y,
                tile_bytes,
                {},
                {.page_id = m * CT_ARG("h_tiles") + first_column + t});
        }
    };

    for (uint32_t slot = 0; slot < a_depth; slot++) {
        Semaphore<> a_free(CT_ARG("sem_a_free") + slot);
        a_collectors.increment(noc, a_free, 1);
    }

    uint32_t a_rows = 0;
    bool a_credited = true;
    uint32_t h_rows = 0;
    uint32_t y_rows = 0;
    uint32_t sent_rows = 0;
    uint32_t landed_rows[RING_CHIPS - 1] = {};
    uint32_t received_rows = 0;
    auto landed_all = [&]() {
        for (uint32_t source = 0; source < peers; source++) {
            if (landed_rows[source] < rows) {
                return false;
            }
        }
        return true;
    };
    while (a_rows < rows ||
           (owner ? (h_rows < rows || y_rows < rows || !landed_all()) : (sent_rows < rows || received_rows < rows))) {
        if (a_rows < rows) {
            const uint32_t slot = a_rows % a_depth;
            if (!a_credited && a.pages_reservable_at_back(CT_ARG("i_tiles"))) {
                Semaphore<> a_free(CT_ARG("sem_a_free") + slot);
                a_collectors.increment(noc, a_free, 1);
                a_credited = true;
            }
            if (a_credited && a_valid[slot].value() >= CT_ARG("a_collectors") * (a_rows / a_depth + 1)) {
                a.push_back(CT_ARG("i_tiles"));
                a_rows++;
                a_credited = a_rows < a_depth;
            }
        }
        if (owner) {
            if (h_rows < rows && h_segment.pages_reservable_at_back(columns)) {
                for (uint32_t t = 0; t < columns; t++) {
                    noc.async_read(
                        h,
                        CoreLocalMem<uint32_t>(h_segment.get_write_ptr() + t * tile_bytes),
                        tile_bytes,
                        {.page_id = h_rows * CT_ARG("h_tiles") + first_column + t},
                        {});
                }
                noc.async_read_barrier();

                h_segment.push_back(columns);
                h_rows++;
            }
            for (uint32_t source = 0; source < peers; source++) {
                if (landed_rows[source] < rows && land_ready[source].value() > landed_rows[source]) {
                    CircularBuffer(CT_ARG("cb_landing") + source).push_back(columns);
                    landed_rows[source]++;
                }
            }
            if (y_rows < rows && y_segment.pages_available_at_front(columns)) {
                const uint32_t m = y_rows;
                bool credited = true;
                for (uint32_t receiver = 0; receiver < peers; receiver++) {
                    credited = credited && (m < depth || credit_y[receiver].value() >= m + 1 - depth);
                }
                if (credited) {
                    write_segment(m, y_segment.get_read_ptr());
                    for (uint32_t offset = 1; offset <= peers; offset++) {
                        const uint32_t port = offset == RING_CHIPS - 1 ? 1 : 0;
                        const uint32_t group = RING_CHIPS - offset;
                        ports[port].push(
                            noc,
                            MessageKind::Data,
                            port == 1 ? 1 : offset,
                            y_segment.get_read_ptr(),
                            segment_bytes,
                            group_x[group],
                            self_y,
                            landing[0] + (m % depth) * segment_bytes,
                            semaphore_address(CT_ARG("sem_land_ready"), 0));
                    }
                    noc.async_write_barrier();
                    y_segment.pop_front(columns);
                    for (uint32_t offset = 1; offset <= peers; offset++) {
                        const uint32_t port = offset == RING_CHIPS - 1 ? 1 : 0;
                        ports[port].push(
                            noc,
                            MessageKind::Increment,
                            port == 1 ? 1 : offset,
                            0,
                            0,
                            group_x[RING_CHIPS - offset],
                            self_y,
                            0,
                            semaphore_address(CT_ARG("sem_credit_partial"), 0));
                    }
                    y_rows++;
                }
            }
        } else {
            if (sent_rows < rows && send.pages_available_at_front(columns) &&
                (sent_rows < depth || credit_partial.value() >= sent_rows + 1 - depth)) {
                const uint32_t owner_chip = (chip + RT_ARG("group")) % chips;
                ports[0].push(
                    noc,
                    MessageKind::Data,
                    RT_ARG("hops"),
                    send.get_read_ptr(),
                    segment_bytes,
                    group_x[0],
                    self_y,
                    landing[peer_index(chip, owner_chip)] + (sent_rows % depth) * segment_bytes,
                    semaphore_address(CT_ARG("sem_land_ready"), peer_index(chip, owner_chip)));
                send.pop_front(columns);
                sent_rows++;
            }
            if (received_rows < rows && land_ready[0].value() > received_rows) {
                write_segment(received_rows, landing[0] + (received_rows % depth) * segment_bytes);
                noc.async_write_barrier();
                ports[0].push(
                    noc,
                    MessageKind::Increment,
                    RT_ARG("hops"),
                    0,
                    0,
                    group_x[0],
                    self_y,
                    0,
                    semaphore_address(CT_ARG("sem_credit_y"), 3 - RT_ARG("group")));
                received_rows++;
            }
        }
    }
    noc.async_write_barrier();
    for (uint32_t port = 0; chips > 1 && port < (owner ? 2 : 1); port++) {
        ports[port].push(noc, MessageKind::End, 0, 0, 0, 0, 0, 0, 0);
    }
    noc.async_atomic_barrier();
}
