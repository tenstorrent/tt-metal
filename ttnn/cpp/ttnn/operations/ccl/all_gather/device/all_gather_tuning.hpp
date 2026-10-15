// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

////////////////////////////////////////////////////////////////
// QUARANTINED TEST HOOK -- delete this file (and its .cpp, the `tuning` member of AllGatherParams, and every
// `params.tuning.` / `tuning.` read in the factories) once the sweep is done, keeping the winning defaults.
// Nothing outside the all_gather op may depend on it.
//
// Overrides come from ONE env var, read when the op's params are built:
//
//   TT_AG_TUNING="key=value,key=value,..."      (unset or empty -> all defaults -> step-1 behaviour)
//
// Keys (every one has a consumer; see APPLY2.md for where):
//   factory          multicast | unicast              force select_program_factory's choice
//   k                1 | auto | <power of 2>          chunks per walk page (1 = the v1 walk; auto = choose_k)
//   xfer             legacy | segments                xfer rule (segments = scatter-segment cap, gather_rules.hpp)
//   rotation         none | absolute | absolute_dir_stagger   lane rotation of full tiles
//   entry            legacy | scatter_aware | one_packet      CB entry sizing (one_packet == scatter_aware)
//   padded_slot      0 | 1                            CB slot = the output's aligned page (padded output, 1 chunk/page)
//   run_cap          <bytes>                          run cap (0 = no cap)
//   workers          <n>                              unicast workers_per_dir
//   signals          <n>                              unicast data_valid signals per stripe
//   mcast_cb_depth   <n>                              multicast CB depth (>= 2)
//   packet           <bytes>                          packet size override
//
// Unknown keys and malformed values TT_THROW, so a typo cannot silently run the default.
// The struct is part of AllGatherParams' attribute list, hence part of the program-cache hash: two runs with
// different overrides never share a cached program.
//
// Self-contained on purpose (no all_gather_geometry.hpp include): all_gather_device_operation_types.hpp includes
// this, and all_gather_geometry.hpp includes this for the policy enums.
////////////////////////////////////////////////////////////////

#include <cstdint>
#include <optional>
#include <string_view>
#include <tuple>

namespace ttnn::operations::ccl {

enum class AllGatherAlgorithm : uint8_t { Multicast, Unicast };

// Lane rotation of full tiles (gather_rules.hpp rot_target / rot_for_target).
//   None               rot = 0: the v1 order.
//   Absolute           slice i of n starts its full tiles on absolute lane (i * stride) / n.
//   AbsoluteDirStagger as Absolute, with unicast's backward direction offset by max(1, stride / (2n)).
enum class RotationMode : uint8_t { None, Absolute, AbsoluteDirStagger };

// How a CB entry is sized (WalkGeometry::make).
//   Legacy       -- step 1's formula: whole packet loads rounded to whole pages (chunks_per_group of the k = 1
//                   geometry) times packets_per_cb_entry, clamped by L1. At k > 1 that group is 1 (k > 1 is
//                   Matched only), so the entry is not rounded to whole walk pages -- rounding would cap a
//                   packet at the k = 1 payload and undo k.
//   ScatterAware -- one packet: min(xfer, kNocScatterWriteMaxChunks * shortest output run), never rounded.
enum class EntryPolicy : uint8_t { Legacy, ScatterAware };

// How xfer (chunks per column) is chosen (WalkGeometry::make).
//   Legacy   -- step 1: chunks_per_transfer(packet, slot, run_cap) if the output is packed, else 1.
//   Segments -- chunks_per_transfer, then capped so a column holds <= 4 scatter segments
//               (gather_rules.hpp xfer_for_segments): unpacked outputs get in-page runs instead of xfer 1.
enum class XferPolicy : uint8_t { Legacy, Segments };

struct Tuning {
    std::optional<AllGatherAlgorithm> force_factory;
    uint32_t walk_k = 1;  // 0 = auto (choose_k); else a power of 2
    XferPolicy xfer_policy = XferPolicy::Legacy;
    RotationMode rotation = RotationMode::None;
    EntryPolicy entry_policy = EntryPolicy::Legacy;
    bool padded_slot = false;
    std::optional<uint32_t> run_cap_bytes;
    std::optional<uint32_t> workers_per_dir;
    std::optional<uint32_t> signals_per_stripe;
    std::optional<uint32_t> mcast_cb_depth;
    std::optional<uint32_t> packet_size_override;

    static constexpr uint32_t kAutoK = 0;

    // User-declared so Tuning is NOT an aggregate: an aggregate is also ttsl::concepts::Reflectable, and then
    // reflection.hpp's attribute_names and auto-reflection specializations both match (ambiguous to_json_t).
    Tuning() = default;

    static Tuning from_env();
    static Tuning parse(std::string_view spec);  // from_env() minus the getenv, for tests

    // TT_FATALs naming the first key that is set but not consumed by the op yet. Every key is wired as of
    // step 2, so this accepts everything; kept as the place a future, not-yet-wired key is rejected.
    void require_only_wired_keys() const;

    bool operator==(const Tuning&) const = default;

    // Same style as AllGatherParams: every field in both lists, or two configs share a cached program.
    static constexpr auto attribute_names = std::forward_as_tuple(
        "force_factory",
        "walk_k",
        "xfer_policy",
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
            walk_k,
            xfer_policy,
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
