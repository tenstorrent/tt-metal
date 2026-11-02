// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Slot donation (kernels/ring_joint_ksplit.hpp) on the K split's dense layout: the cores that own no unit pair up as
// spares, and every home head pair gives one slot of its ring walk to a spare pair. A row's K multicast moves at the
// pace of its slowest core, so all homes linked through a row (their head pairs straddle rows) give the same slot and
// skip it together; each such group takes its own slot. The last band's groups take the first slots (the local slice
// is the smallest on most devices and the first band, without the diagonal, the slower one), so the first band gives
// away larger slots. A spare pair walks its items in slot order, each once its slice has arrived, so spares need not
// keep pace with the slot they serve.

#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_ksplit.hpp"

namespace ttnn::operations::transformer::sdpa::ring_joint {

constexpr uint32_t kNoDonatedSlot = 0xFFFFFFFFu;

// A core's interleaved walk (kernels/ring_joint_ksplit.hpp) as slots: slot 0 = iteration 0, slot 1 + kInterleavePieces
// * k + p = piece p of pair k (one step of each of its two iterations), and an unpaired last iteration. A slot's steps
// are consecutive: [slot_first_step, slot_first_step + slot_step_count).
inline uint32_t paired_slot_count(uint32_t ring_iterations) { return ((ring_iterations - 1) / 2) * kInterleavePieces; }

inline uint32_t slot_count(uint32_t ring_iterations) {
    return 1 + paired_slot_count(ring_iterations) + (ring_iterations - 1) % 2;
}

inline uint32_t slot_first_step(uint32_t slot, uint32_t ring_iterations) {
    if (slot == 0) {
        return 0;
    }
    const uint32_t s = slot - 1;
    if (s < paired_slot_count(ring_iterations)) {
        return 1 + (s / kInterleavePieces) * 2 * kInterleavePieces + (s % kInterleavePieces) * 2;
    }
    return 1 + paired_slot_count(ring_iterations) * 2;
}

inline uint32_t slot_step_count(uint32_t slot, uint32_t ring_iterations) {
    return slot == 0 || slot - 1 >= paired_slot_count(ring_iterations) ? 1 : 2;
}

// A slot's steps as the kernels' donation args pack them.
inline uint32_t donation_steps(uint32_t slot, uint32_t ring_iterations) {
    return slot_first_step(slot, ring_iterations) | slot_step_count(slot, ring_iterations) << 8;
}

struct DonationVisit {
    uint32_t slot = 0;
    uint32_t unit = 0;
    uint32_t band = 0;
    uint32_t home = 0;  // core index
};

struct DonationPlan {
    bool enabled = false;
    std::vector<uint32_t> donated_slot;              // per core: its group's slot, or kNoDonatedSlot
    std::vector<uint32_t> visitor;                   // per home core: the spare that takes its donated slot
    std::vector<std::vector<DonationVisit>> visits;  // per core: a spare's visits in order
    std::vector<uint32_t> spare_rank, spare_peer;    // per core: a spare's rank and peer in its pair
    std::vector<uint32_t> row_homes;                 // per row: leading cores that own a unit (the multicast)
};

// Core (x, y) in band y / rows_per_split owns unit (y % rows_per_split) * grid_x + x (the factory's dense layout);
// a head's two Q chunks are units 2h and 2h + 1.
inline DonationPlan make_donation_plan(
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t rows_per_split,
    uint32_t ksplit_count,
    uint32_t total_q_chunks,
    uint32_t q_chunks_per_head,
    uint32_t ring_iterations,
    bool arrival_order = false) {
    DonationPlan plan;
    const uint32_t num_cores = grid_x * grid_y;
    const uint32_t slots = slot_count(ring_iterations);
    if (ksplit_count < 2 || q_chunks_per_head != 2 || total_q_chunks % 2 != 0 || slots < 2 ||
        rows_per_split * grid_x < total_q_chunks || ksplit_count * rows_per_split > grid_y) {
        return plan;
    }
    auto owns_unit = [&](uint32_t core) {
        const uint32_t y = core / grid_x;
        return y / rows_per_split < ksplit_count && (y % rows_per_split) * grid_x + core % grid_x < total_q_chunks;
    };
    auto home_core = [&](uint32_t unit, uint32_t band) {
        return (band * rows_per_split + unit / grid_x) * grid_x + unit % grid_x;
    };
    std::vector<uint32_t> spares;
    for (uint32_t core = 0; core < num_cores; ++core) {
        if (!owns_unit(core)) {
            spares.push_back(core);
        }
    }
    // Homes lead every row, so each row's multicast stays one rectangle that leaves the spares out.
    plan.row_homes.assign(grid_y, 0);
    for (uint32_t y = 0; y < grid_y; ++y) {
        uint32_t width = 0;
        while (width < grid_x && owns_unit(y * grid_x + width)) {
            ++width;
        }
        for (uint32_t x = width; x < grid_x; ++x) {
            if (owns_unit(y * grid_x + x)) {
                return plan;
            }
        }
        plan.row_homes[y] = width;
    }
    const uint32_t heads = total_q_chunks / 2;
    const uint32_t spare_pairs = static_cast<uint32_t>(spares.size()) / 2;
    const uint32_t items = heads * ksplit_count;
    if (spares.size() % 2 != 0 || spare_pairs == 0 || spare_pairs * (slots - 1) < items ||
        (items + spare_pairs - 1) / spare_pairs > kDonationMaxVisits) {
        return plan;
    }
    // Row groups per band: rows joined by a head pair that straddles them.
    std::vector<uint32_t> group_of_row(rows_per_split);
    std::iota(group_of_row.begin(), group_of_row.end(), 0);
    for (uint32_t h = 0; h < heads; ++h) {
        const uint32_t a = (2 * h) / grid_x;
        const uint32_t b = (2 * h + 1) / grid_x;
        const uint32_t ga = group_of_row[a];
        const uint32_t gb = group_of_row[b];
        for (auto& g : group_of_row) {
            if (g == gb) {
                g = ga;
            }
        }
    }
    std::vector<uint32_t> group_ids(group_of_row);
    std::sort(group_ids.begin(), group_ids.end());
    group_ids.erase(std::unique(group_ids.begin(), group_ids.end()), group_ids.end());
    const uint32_t groups_per_band = static_cast<uint32_t>(group_ids.size());
    if (groups_per_band * ksplit_count > slots - 1) {
        return plan;
    }
    // Items: within a group, heads in order; spares walk them in slot order. With arrival_order (a rotating KV window,
    // which reuses a gathered slot once its last reader moves on) a spare takes each gathered item about when its
    // slice arrives, slot s some s slots into the walk, and fills the waits with local items.
    struct Item {
        uint32_t head, band, slot;
    };
    std::vector<Item> item_list;
    plan.donated_slot.assign(num_cores, kNoDonatedSlot);
    for (uint32_t band = 0; band < ksplit_count; ++band) {
        for (uint32_t g = 0; g < groups_per_band; ++g) {
            const uint32_t slot = (ksplit_count - 1 - band) * groups_per_band + g;
            for (uint32_t h = 0; h < heads; ++h) {
                if (group_of_row[(2 * h) / grid_x] != group_ids[g]) {
                    continue;
                }
                item_list.push_back({h, band, slot});
                for (uint32_t r = 0; r < 2; ++r) {
                    plan.donated_slot[home_core(2 * h + r, band)] = slot;
                }
            }
        }
    }
    std::sort(item_list.begin(), item_list.end(), [](const Item& a, const Item& b) {
        return a.slot != b.slot ? a.slot < b.slot : (a.band != b.band ? a.band < b.band : a.head < b.head);
    });
    plan.visits.assign(num_cores, {});
    plan.visitor.assign(num_cores, 0);
    plan.spare_rank.assign(num_cores, 0);
    plan.spare_peer.assign(num_cores, 0);
    for (uint32_t p = 0; p < spare_pairs; ++p) {
        for (uint32_t r = 0; r < 2; ++r) {
            plan.spare_rank[spares[2 * p + r]] = r;
            plan.spare_peer[spares[2 * p + r]] = spares[2 * p + 1 - r];
        }
    }
    std::vector<std::vector<Item>> pair_items(spare_pairs);
    for (uint32_t j = 0; j < item_list.size(); ++j) {
        pair_items[j % spare_pairs].push_back(item_list[j]);
    }
    if (arrival_order) {
        // Visit t starts about t slots into the walk: take the earliest gathered item that has arrived by then, else a
        // local one, else the earliest gathered one (and wait for it).
        for (auto& items : pair_items) {
            std::vector<Item> local, remote, order;
            for (const Item& item : items) {
                (item.slot == 0 ? local : remote).push_back(item);
            }
            size_t next_local = 0, next_remote = 0;
            for (uint32_t t = 0; t < items.size(); ++t) {
                const bool remote_ready = next_remote < remote.size() && remote[next_remote].slot <= t;
                if (remote_ready || next_local == local.size()) {
                    order.push_back(remote[next_remote++]);
                } else {
                    order.push_back(local[next_local++]);
                }
            }
            items = std::move(order);
        }
    }
    for (uint32_t p = 0; p < spare_pairs; ++p) {
        for (const Item& item : pair_items[p]) {
            for (uint32_t r = 0; r < 2; ++r) {
                const uint32_t unit = 2 * item.head + r;
                plan.visits[spares[2 * p + r]].push_back({item.slot, unit, item.band, home_core(unit, item.band)});
                plan.visitor[home_core(unit, item.band)] = spares[2 * p + r];
            }
        }
    }
    plan.enabled = true;
    return plan;
}

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
