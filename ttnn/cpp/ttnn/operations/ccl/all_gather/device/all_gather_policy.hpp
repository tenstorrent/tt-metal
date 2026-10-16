// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <umd/device/types/arch.hpp>

// Internal to the all_gather op: include from .cpp files only (it is not an installed header).

////////////////////////////////////////////////////////////////
// all_gather's per-arch policy: every tuned value either factory uses, chosen from measurements, in one place.
//
//   prefers_unicast  -- route_all_gather's heuristic: does unicast beat multicast here?
//   tune_unicast     -- the store-and-forward factory's walk, CB, workers, mux and data_valid cadence.
//   tune_multicast   -- the multicast factory's walk and CB.
//
// Two tune functions, not one: the knob sets differ (unicast has workers / mux slots / signals and a fixed CB depth;
// multicast tunes the CB depth), and they run at different points (unicast before its core-grid shrink, at the
// requested link count; multicast after its shrink, at the link count it runs).
//
// Every input is a hashed op attribute or derives from one (output spec, topology, links, device count) or is a
// per-mesh device constant (arch), so a program-cache entry always matches the policy it was built with. Nothing
// here may depend on unhashed state (e.g. free L1).
////////////////////////////////////////////////////////////////

namespace ttnn::operations::ccl {

struct WalkPolicy {
    static constexpr uint32_t kAutoK = 0;
    uint32_t k = 1;  // chunks per walk page: kAutoK = choose_k (only Matched, unpadded pages), else 1
    // How a CB entry is sized (WalkGeometry::make). By default: SharedPolicy::packets_per_cb_entry packet loads,
    // rounded to whole input (split) or output (concat) pages of the k = 1 geometry, clamped by L1 (at k > 1 that
    // group is 1 -- k > 1 is Matched only -- so the entry is not rounded back up to a page).
    // With this set, a stripe no longer than the walk stride (walk_pages_per_stripe <= stride: every output lane run
    // is one walk page, so the 4-segment scatter cap binds) gets a one-packet entry instead: min(xfer,
    // gather_scatter_max_chunks * the shortest output run), never rounded, so the writer never waits on segments a
    // packet cannot carry. packets_per_cb_entry does not apply to those.
    bool one_packet_short_stripe = false;
    // CB slot = the output's padded aligned page where one chunk fills an output page whose aligned page is
    // padded (relay_needs_padded_slot); a no-op elsewhere. Unicast's relay needs it.
    bool padded_slot = false;
};

// What the policies key on. All sizes are in walk pages / bytes of the k = 1 geometry, so k never moves a tier.
struct PolicyInputs {
    tt::ARCH arch = tt::ARCH::Invalid;
    bool is_ring = false;         // unicast: the gather axis wraps; multicast: any active axis wraps
    uint32_t num_devices = 0;     // devices in the collective
    uint32_t num_links = 0;       // unicast: requested links on the axis; multicast: links it runs
    uint64_t per_link_bytes = 0;  // packed bytes of the gathered output per link (at num_links)
    uint32_t stripe_pages = 0;    // walk pages this device contributes per output row
};

struct SharedPolicy {
    WalkPolicy walk;
    uint32_t run_cap_bytes = 0;         // cap on one transfer's bytes (0 = none); also choose_k's budget
    uint32_t packets_per_cb_entry = 1;  // packet loads per CB entry (not for a one-packet entry)
    uint32_t cb_depth = 2;              // CB entries; 2 is the floor (the reader runs one entry ahead)
};

struct UnicastPolicy {
    SharedPolicy shared;
    uint32_t workers_per_dir = 1;       // > 1 shares a fabric mux core per direction per link
    uint8_t mux_slots_per_channel = 2;  // buffers per mux channel (1 or 2)
    uint32_t signals_per_stripe = 2;    // data_valid signals per relayed stripe
};

struct MulticastPolicy {
    SharedPolicy shared;
};

bool prefers_unicast(const PolicyInputs& in);
UnicastPolicy tune_unicast(const PolicyInputs& in);
MulticastPolicy tune_multicast(const PolicyInputs& in);

}  // namespace ttnn::operations::ccl
