// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

////////////////////////////////////////////////////////////////
// QUARANTINED TEST HOOK -- delete this file (and its .cpp, the `tuning` member of AllGatherParams, and every
// `params.tuning.` read) once the sweep is done. Nothing outside the all_gather op may depend on it.
//
// Overrides come from ONE env var, read when the op's params are built:
//
//   TT_AG_TUNING="key=value,key=value,..."      (unset or empty -> all defaults -> today's behaviour)
//
// Keys:
//   factory          multicast | unicast              force select_program_factory's choice
//   rotation         none | absolute | absolute_dir_stagger
//   entry            legacy | scatter_aware           CB entry sizing (EntryPolicy)
//   padded_slot      0 | 1
//   run_cap          <bytes>                          run cap (0 = no cap)
//   workers          <n>                              unicast workers_per_dir
//   signals          <n>                              unicast data_valid signals per stripe
//   mcast_cb_depth   <n>                              multicast CB depth
//   packet           <bytes>                          packet size override
//
// Unknown keys and malformed values TT_THROW, so a typo cannot silently run the default.
// Keys parsed but not yet consumed by the op are rejected by require_only_wired_keys(), so setting one cannot
// silently run the default either.
// The struct is part of AllGatherParams' attribute list, hence part of the program-cache hash: two runs with
// different overrides never share a cached program.
//
// Self-contained (std only): AllGatherParams, an installed header, includes this one.
////////////////////////////////////////////////////////////////

#include <cstdint>
#include <optional>
#include <string_view>
#include <tuple>

namespace ttnn::operations::ccl {

// How a CB entry is sized (WalkGeometry::make).
//   Legacy       -- a whole number of packet loads, rounded to whole pages (chunks_per_group), times
//                   packets_per_cb_entry, clamped by L1.
//   ScatterAware -- one transfer's worth, but no more than NOC_SCATTER_WRITE_MAX_CHUNKS of the shortest output
//                   run: min(xfer, 4 * min_output_run).
enum class EntryPolicy : uint8_t { Legacy, ScatterAware };

enum class AllGatherAlgorithm : uint8_t { Multicast, Unicast };
enum class RotationMode : uint8_t { None, Absolute, AbsoluteDirStagger };

struct Tuning {
    std::optional<AllGatherAlgorithm> force_factory;
    RotationMode rotation = RotationMode::None;
    EntryPolicy entry_policy = EntryPolicy::Legacy;
    bool padded_slot = false;
    std::optional<uint32_t> run_cap_bytes;
    std::optional<uint32_t> workers_per_dir;
    std::optional<uint32_t> signals_per_stripe;
    std::optional<uint32_t> mcast_cb_depth;
    std::optional<uint32_t> packet_size_override;

    // User-declared so Tuning is NOT an aggregate: an aggregate is also ttsl::concepts::Reflectable, and then
    // reflection.hpp's attribute_names and auto-reflection specializations both match (ambiguous to_json_t).
    Tuning() = default;

    static Tuning from_env();
    static Tuning parse(std::string_view spec);  // from_env() minus the getenv, for tests

    // TT_FATALs naming the first key that is set but not consumed by the op yet (only `factory` is wired).
    void require_only_wired_keys() const;

    bool operator==(const Tuning&) const = default;

    // Same style as AllGatherParams: every field in both lists, or two configs share a cached program.
    static constexpr auto attribute_names = std::forward_as_tuple(
        "force_factory",
        "rotation",
        "entry_policy",
        "padded_slot",
        "run_cap_bytes",
        "workers_per_dir",
        "signals_per_stripe",
        "mcast_cb_depth",
        "packet_size_override");
    auto attribute_values() const {
        return std::forward_as_tuple(
            force_factory,
            rotation,
            entry_policy,
            padded_slot,
            run_cap_bytes,
            workers_per_dir,
            signals_per_stripe,
            mcast_cb_depth,
            packet_size_override);
    }
};

}  // namespace ttnn::operations::ccl
