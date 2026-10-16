// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_policy.hpp"

#include <algorithm>
#include <array>

#include <tt_stl/assert.hpp>

namespace ttnn::operations::ccl {

namespace {

constexpr uint64_t KiB = 1024;
constexpr uint64_t MiB = 1024 * KiB;

// The checks every policy must pass, whatever the tier tables say.
void check_shared(const SharedPolicy& p, const char* who) {
    TT_FATAL(
        p.walk.k == 1 || p.walk.k == WalkPolicy::kAutoK, "all_gather {} policy: k={} (want 1 or auto)", who, p.walk.k);
    TT_FATAL(p.packets_per_cb_entry >= 1, "all_gather {} policy: packets_per_cb_entry=0", who);
    TT_FATAL(p.cb_depth >= 2, "all_gather {} policy: cb_depth={} (the reader runs one entry ahead)", who, p.cb_depth);
}

// BH ring workers_per_dir by total output bytes (with a long stripe; see tune_unicast).
struct WorkersTier {
    uint64_t min_total_bytes;
    uint32_t workers;
};
constexpr std::array<WorkersTier, 4> kBhRingWorkers = {{
    {0, 1},
    {64 * KiB, 2},
    {1536 * KiB, 3},
    {16 * MiB, 4},
}};
constexpr bool tiers_sorted() {
    for (size_t i = 1; i < kBhRingWorkers.size(); ++i) {
        if (kBhRingWorkers[i].min_total_bytes <= kBhRingWorkers[i - 1].min_total_bytes) {
            return false;
        }
    }
    return kBhRingWorkers[0].min_total_bytes == 0;
}
static_assert(tiers_sorted(), "kBhRingWorkers must start at 0 and be strictly increasing");

uint32_t bh_ring_workers(uint64_t total_output_bytes) {
    uint32_t workers = 1;
    for (const auto& tier : kBhRingWorkers) {
        if (total_output_bytes >= tier.min_total_bytes) {
            workers = tier.workers;
        }
    }
    return workers;
}

}  // namespace

////////////////////////////////////////////////////////////////
// Routing heuristic
////////////////////////////////////////////////////////////////

bool prefers_unicast(const PolicyInputs& in) {
    switch (in.arch) {
        case tt::ARCH::WORMHOLE_B0: {
            // Multicast's packets are copied onward by the fabric; unicast re-reads and re-sends at every hop,
            // but its relay splits the sending across devices. So hop count decides, and device count sets hop
            // count -- non-monotonically:
            //   ring       always multicast; the fabric does the relay for free.
            //   2 devices  one hop, so unicast never relays while multicast still pays its barrier fan-in and
            //              packet setup.
            //   3-6        multicast, its edge shrinking as N grows.
            //   7+         near tie, so size decides: unicast's split sending pays once the link saturates.
            // Calibrated on T3000 at 2-8 devices; the shrinking-edge trend makes the size test the safer
            // extrapolation above 8.
            if (in.is_ring) {
                return false;
            }
            if (in.num_devices <= 2) {
                return true;
            }
            if (in.num_devices <= 6) {
                return false;
            }
            return in.per_link_bytes >= 3'000'000;
        }
        case tt::ARCH::BLACKHOLE: {
            // Multicast buys the hops unicast's store-and-forward relay must pay (N/2 on a ring, N-1 on a
            // line), so the crossover moves sharply with device count: at 4 devices a ring is 2 hops and
            // unicast wins outright; at 8, multicast takes a wide band at the small end. Neither page size nor
            // stripe length moves it.
            // Linear in device count is a two-point fit (4 and 8 devices), not a law: re-measure before
            // trusting it beyond 8. Both thresholds floor at zero, which is also right for the 2- and 3-device
            // cases they extrapolate to -- one or two hops of relay is never worth a multicast.
            // Re-checked with the shipped policies (unicast k = auto and one-packet entries, multicast k = auto
            // without a run cap, each pinned) against the routed default:
            // Data:     <R3: reports 2026_10_16_01_35_27 (tensor-size ladder 8K-100M, tile) and 2026_10_16_01_43_23
            //            (page ladder 64 B-8 KB, row-major), BH 1x8, 2 links, packet 15232, 20 iters, ring and line>
            // Result:   both 8-device thresholds sit inside the measured tie bands (ring: no decisive multicast win
            //           anywhere, unicast decisive from 5.2 MB/link; line: multicast decisive to 0.92 MB/link, tie
            //           1.0-2.6 MB/link, unicast decisive from 5.2 MB/link). In every decided cell the routed factory
            //           was the faster one (ring 12/12, line 17/18; the one miss, line 10M, was unicast's stripe-10
            //           entry, fixed in tune_unicast below). Page size does not move the margin (flat 64 B-8 KB).
            // Validity: 8 devices x 2 links; the 4-device points come from the original two-machine fit.
            const int64_t dev = static_cast<int64_t>(in.num_devices);
            const int64_t threshold_kb = std::max<int64_t>(
                0,
                in.is_ring ? 320 * (dev / 2 - 2)      // 4 dev -> 0,      8 dev -> 640 KB
                           : 384 + 544 * (dev - 4));  // 4 dev -> 384 KB, 8 dev -> 2560 KB
            return in.per_link_bytes >= static_cast<uint64_t>(threshold_kb) * KiB;
        }
        default: return false;  // uncalibrated arch
    }
}

////////////////////////////////////////////////////////////////
// Unicast (store-and-forward)
////////////////////////////////////////////////////////////////

UnicastPolicy tune_unicast(const PolicyInputs& in) {
    UnicastPolicy p;
    // The padded slot is not optional here: a one-chunk-per-page output whose aligned page is padded can only be
    // relayed with slot == out_aligned_page (relay_needs_padded_slot). On always -- it is a no-op wherever the
    // output's aligned page is the chunk -- so the 2-device / no-relay case walks the same way.
    p.shared.walk.padded_slot = true;
    // Two entries: one filling while the other drains.
    p.shared.cb_depth = 2;

    if (in.arch == tt::ARCH::WORMHOLE_B0) {
        // Calibrated on T3000 at 2, 4 and 8 devices, at walk k = 1 with three-packet entries. k = auto and the
        // one-packet entry were measured on Blackhole only (see below), and the one-packet entry replaces this
        // calibrated three-packet one, so both stay off on WH until measured there.
        // <WH: k=auto and one-packet entries vs these values on T3000>
        //
        // workers_per_dir: a second worker needs a fabric mux -- 6 cores per link instead of 2, plus a
        // hop per packet -- so take it only where it is not slower, i.e. while the op is latency-bound.
        // A long stripe outlasts a transfer many times over, so a worker's runs stay inside one row and
        // its writes land sequentially at the destination; short stripes straddle a row edge on every
        // transfer, which is where the extra worker starts paying.
        const bool long_stripe = in.stripe_pages >= 64;
        // Keyed on this device's own share of the link, not the whole gathered output: that share is what
        // actually transfers, and the boundaries below sit at the same per-device figure at 2, 4 and 8
        // devices. Stated against the total they would only be right at 8.
        const uint64_t device_bytes_per_link = in.per_link_bytes / std::max(1u, in.num_devices);
        const bool small = device_bytes_per_link <= 64 * KiB;
        const bool long_stripe_wins = long_stripe && (in.is_ring || device_bytes_per_link <= 512 * KiB);
        p.workers_per_dir = (small || long_stripe_wins) ? 1 : 2;
        // Three packets per entry: multicast wants 1 here; every cell was swept on its own.
        p.shared.packets_per_cb_entry = 3;
        // mux_slots_per_channel: two slots let a worker stage its next packet while the mux forwards the
        // last. On a ring at scale that only interleaves the two workers' packets more finely at the
        // receiver, scattering its DRAM writes, so one slot wins there; a line, whose per-hop relay is
        // already serialised, keeps two. Ring only, and a ring cannot be shortened on a T3000 (the
        // wrap-around link needs all 8), so this one stays in whole-output terms.
        p.mux_slots_per_channel = (in.is_ring && in.per_link_bytes >= 2 * MiB) ? 1 : 2;
        // No run cap: the best run length lands at the hardware ceiling (7616 B), so any value settable
        // here is already above it and would only cost payload.
    } else if (in.arch == tt::ARCH::BLACKHOLE) {
        // workers_per_dir and run_cap_bytes were first calibrated on two machines, 8 devices x 2 links and
        // 4 devices x 4 links with the link count also forced down to 2 and 1, at walk k = 1 with the packet-load
        // entry. Page size moves none of the boundaries; stripe length and link count do.
        //
        // workers_per_dir: each extra worker feeds a link harder but costs a core and, past one, a mux
        // core plus a NOC hop per packet. How far that pays scales with link count, because workers are
        // per-link while DRAM is shared -- so at 4 links a fourth worker regresses, while at 2 links it
        // is worth 1.4-2.2% once the output passes ~16 MB.
        // The scaling variable differs by topology:
        //  - Ring: total output bytes (kBhRingWorkers). The boundary sits at the same total volume at every link
        //    count measured (the per-link figure moves with links, the total does not).
        //  - Line: per-link bytes divided by device count. A line's relay crosses N-1 hops, so its
        //    per-device relay load -- and with it the point where another worker pays -- scales with N.
        //    This one is a two-machine fit (4 and 8 devices), not a measured law.
        // Re-checked on top of k = auto and the one-packet entry:
        // Data:     <R2: report 2026_10_16_00_58_15 (R2a: workers=1..4 vs these tiers), BH 1x8, 2 links, packet
        //            15232, 20 iters, DRAM interleaved bf16, ring and line; shapes 512K, 5M, 25M, p2K, p8K>
        // Result:   the tiers pick the best measured count in every cell (ring 1 MB -> 2, 10.5 MB -> 3, 16 MiB and
        //           50 MB -> 4; line 2 throughout). Every explicit alternative is level or slower: 1 worker +5% to
        //           +131%, line 3-4 workers +1% to +15%. Softest spot: ring at exactly 16 MiB, where 3 workers is
        //           2.9% faster than 4 (CIs apart, under the 5% bar).
        // Validity: 8 devices x 2 links only; the 4-link and 4-device points are from the first calibration.
        const uint64_t total_output_bytes = in.per_link_bytes * in.num_links;
        // A stripe shorter than a transfer straddles a row edge on nearly every send, so the extra workers
        // cannot land sequential writes and stop paying.
        const bool long_stripe = in.stripe_pages >= 8;
        if (in.is_ring) {
            p.workers_per_dir = bh_ring_workers(total_output_bytes);
            if (!long_stripe) {
                p.workers_per_dir = std::min(p.workers_per_dir, 2u);
            }
        } else {
            // A line wants two workers over essentially its whole useful range: three loses 3-8% from
            // 1 MB per link up, and below ~256 KB per link the choice is inside the noise.
            const uint64_t dev = std::max(1u, in.num_devices);
            p.workers_per_dir = in.per_link_bytes < (640 * KiB / dev) ? 1u : 2u;
        }
        // run_cap_bytes: capping a run costs packet fill but stops the walk parking in one DRAM bank. Every link
        // runs its own 2 x workers_per_dir readers against one shared DRAM, so the more links, the shorter each
        // run has to be to keep the banks spread -- but only once there are enough links to create that pressure.
        // Data:     <R2: report 2026_10_16_00_58_15 (R2a: run_cap=4096 / 8192 vs no cap), 2 links>
        // Result:   at 2 links any cap is a large loss: 4096 B +21% to +145% in every cell, 8192 B +30% to +44% on
        //           every shape >= 5 MB (and it undoes k = auto on row-major pages).
        // Validity: the 4-link half of the rule (+22% on a 4-device ring in the first calibration, before the
        //           output-alignment and init-barrier fixes) is untested on this machine; it is kept as calibrated.
        const bool cap_pays = in.num_links >= 4 && total_output_bytes >= 8 * MiB;
        p.shared.run_cap_bytes = cap_pays ? (16384u / std::max(1u, in.num_links)) : 0u;
        // mux_slots_per_channel stays 2: one slot wins only on an even split, and loses more when the split
        // is uneven than it wins there, so the even case is not worth special-casing.

        // k = auto (choose_k: the smallest k whose chunk fills >= 90% of a packet).
        // Data:     <R1: report 2026_10_16_00_24_25 (R1a: factory=unicast vs factory=unicast,k=auto), BH 1x8,
        //            2 links, packet 15232, 20 iters, DRAM interleaved bf16, ring and line>
        // Result:   row-major 8 KB pages -30.5% line / -22.7% ring, 4 KB pages -9.2% / -6.0%; tile shapes
        //           (2 KB pages, k stays 1) and 64 B-2 KB pages within noise.
        // Validity: k only moves Matched outputs with unpadded pages above ~packet / 2; with a run cap (4 links)
        //           the cap is choose_k's budget, so k stays near 1 there.
        p.shared.walk.k = 1;  // COMMIT 1: today's value; commit 2 ships kAutoK
        // CB entry. A stripe no longer than the walk stride makes every output lane run one walk page, so a packet
        // carries at most 4 of them (the scatter cap binds): there the entry is one packet, so the writer never
        // waits on chunks a packet cannot carry (WalkPolicy::one_packet_short_stripe). Longer stripes keep packet-
        // load entries.
        // Data:     <R1: report 2026_10_16_00_24_25 (R1a: entry=one_packet, every stripe)>; <R3: report
        //            2026_10_16_01_35_27 (factory=unicast,k=auto,entry=one_packet vs the routed default)>; the
        //            stripe-plateau analysis in ag_10m_dip_analysis.md
        // Result:   one packet wins at stripe <= stride (8 DRAM banks): stripe 8 (512K) -15.8% line / -7.7% ring,
        //           stripe 2 (2M) -12.5% / -8.0% (R1); it loses at stripe 10 (> stride): 10M +8.0% ring / +11.9%
        //           line, 2.6M +6.8% / +6.3% (R3); stripes 20 and 32 and every row-major shape within noise (R1).
        p.shared.walk.one_packet_short_stripe = false;  // COMMIT 1: today's entry; commit 2 ships true
        // packets_per_cb_entry for the packet-load entries: one, except in a narrow stripe band. A two-packet entry
        // halves the reader/writer handshake count but leaves the writer trailing an extra packet behind, and that
        // lag costs 1-4% at almost every stripe length -- measured at 2, 4, 5, 8, 20, 32 and 64 chunks. Between 9
        // and 15 it inverts hard: +16% on a ring and +23-27% on a line at stripes 10 and 12, with stripes 8 and 16
        // within half a percent either way. A packet holds 7 chunks of a 2 KB page, so this band is where a
        // one-packet entry leaves its second entry roughly half empty on every stripe while a two-packet entry
        // covers the stripe in one. Measured band, not a derived one: the boundaries are where the sign flips
        // (first calibration, k = 1). R3's stripe-10 shapes confirm it against the one-packet entry above.
        const bool half_empty_second_entry = in.stripe_pages >= 9 && in.stripe_pages <= 15;
        p.shared.packets_per_cb_entry = half_empty_second_entry ? 2u : 1u;
    }
    // Any other arch: uncalibrated, so the plain defaults above (1 worker, k = 1, one-packet-load entries).

    // data_valid signals per relayed stripe: 2, so the downstream relay starts at the halfway point without paying
    // for a signal per CB entry.
    // Data:     <R2: report 2026_10_16_00_58_15 (R2a: signals=1 / 4 vs 2), BH 1x8, 2 links, k = auto and one-packet
    //            entries on every stripe>
    // Result:   both alternatives only lose: 1 signal +8% to +20% on every line shape; 4 signals +5% to +11% on
    //           ring (and +26% on line 512K). The first calibration (k = 1) had found the same optimum.
    // Validity: BH; WH keeps the same value from its own calibration.
    p.signals_per_stripe = 2;

    check_shared(p.shared, "unicast");
    TT_FATAL(p.shared.walk.padded_slot, "all_gather unicast policy: the relay needs the padded slot");
    TT_FATAL(p.workers_per_dir >= 1, "all_gather unicast policy: workers_per_dir=0");
    TT_FATAL(p.signals_per_stripe >= 1, "all_gather unicast policy: signals_per_stripe=0");
    TT_FATAL(
        p.mux_slots_per_channel == 1 || p.mux_slots_per_channel == 2,
        "all_gather unicast policy: mux_slots_per_channel={} (want 1 or 2)",
        p.mux_slots_per_channel);
    return p;
}

////////////////////////////////////////////////////////////////
// Multicast
////////////////////////////////////////////////////////////////

MulticastPolicy tune_multicast(const PolicyInputs& in) {
    MulticastPolicy p;
    // Multicast never reads its output back, so the padded slot buys it nothing: padded outputs walk one chunk per
    // transfer instead (xfer 1 on an unpacked output).
    p.shared.walk.padded_slot = false;

    if (in.arch == tt::ARCH::WORMHOLE_B0) {
        // The defaults are WH's calibrated values:
        // - cb_depth 2: depth is also the reader's in-flight trid count, but deeper is perf-neutral here, so
        //   keep the L1.
        // - packets_per_cb_entry 1: packing several packets per entry amortises the reader/sender CB
        //   handshake, which is what the store-and-forward writer wants -- but here the reader sends the
        //   packets itself, off transaction ids and with no per-entry flush to stall on, so one packet load per
        //   entry just pipelines finer and wins on a line and on small row-major pages.
        // - No run cap: the best run length lands at the hardware ceiling (7616 B), so any value settable
        //   here is already above it and would only cost payload.
        // - k = 1: k = auto was measured on Blackhole only. <WH: k=auto vs k=1 on T3000>
        p.shared.cb_depth = 2;
        p.shared.packets_per_cb_entry = 1;
        p.shared.run_cap_bytes = 0;
    } else if (in.arch == tt::ARCH::BLACKHOLE) {
        // cb_depth 3: first calibrated at 8 devices x 2 links, and at 1, 2 and 4 links (every other depth inside
        // the run-to-run spread).
        // Data:     <R2: report 2026_10_16_01_12_19 (R2b: mcast_cb_depth=2/4/6 vs 3), BH 1x8, 2 links, packet 15232,
        //            20 iters, DRAM interleaved bf16, ring and line; shapes 131K, 512K, 2M, p2K, p8K>
        // Result:   every depth within +-2.7% on every shape from 2M up and on row-major; no depth beats 3.
        // Validity: 2 would cost nothing measurable and save L1; kept at 3 as calibrated.
        p.shared.cb_depth = 3;
        // packets_per_cb_entry: four amortises the reader/sender handshake and is right almost everywhere,
        // but not on a line carrying a long stripe: there the sender is already serialised per hop, and a
        // four-packet entry just delays the first send behind three more packet loads. A ring, sending in
        // both directions at once, wants the amortisation even at a long stripe, so the rule is line-only.
        // The stripe boundary does not move with volume.
        // ...and only while the link count is low. With 4 links the per-hop send is no longer the
        // serialising step, so the amortisation a four-packet entry buys comes back.
        const bool long_stripe = in.stripe_pages >= 32;
        const bool few_links = in.num_links <= 2;
        p.shared.packets_per_cb_entry = (!in.is_ring && long_stripe && few_links) ? 1u : 4u;
        // No run cap, and k = auto (choose_k fills a whole packet; the run cap used to be its budget).
        // Data:     <R2: report 2026_10_16_01_12_19 (R2b: k=auto,run_cap=0 and run_cap=0 vs the former 8 KB cap
        //            on pages >= 1 KB; R1b, report 2026_10_16_00_39_37, had k=auto under the cap)>
        // Result:   the cap alone changes nothing (all cells within +-4%); k = auto with the cap off makes linear
        //           8 KB row-major pages -19.2% (346.4 -> 279.9 us; -19.0% against the cap-off run alone); ring
        //           and every other shape within noise. Under the cap (R1b) k = auto resolved to 1 everywhere.
        // Validity: 4 KB pages were not in R2b (unicast gained 6-9% there; an expectation, not data).
        p.shared.run_cap_bytes = in.page_bytes >= 1024 ? 8192u : 0u;  // COMMIT 1: today's cap; commit 2: 0
        p.shared.walk.k = 1;                                          // COMMIT 1: commit 2 ships kAutoK
        // Packet-load entries (packets_per_cb_entry) on every stripe: no one-packet entries, not even on the short
        // stripes where unicast uses them (on multicast those were within noise, -1.8% to -4.5%, in R1b).
        // Data:     <R1: report 2026_10_16_00_39_37 (R1b: entry=one_packet); R2: report 2026_10_16_01_12_19 (R2b:
        //            entry=one_packet with mcast_cb_depth=6/8/12)>
        // Result:   one-packet entries cost linear 5M +10.7%, 25M +28.0%, row-major 2 KB +26.4% and 4 KB +20.5%
        //           (R1b); a deeper CB does not help: linear 2 KB +25.4% to +25.9% at depth 6, 8 and 12 (R2b).
        //           Ring within noise.
        p.shared.walk.one_packet_short_stripe = false;
    }
    // Any other arch: uncalibrated, so the plain defaults (cb_depth 2, one-packet-load entries, no cap, k = 1).

    check_shared(p.shared, "multicast");
    return p;
}

}  // namespace ttnn::operations::ccl
