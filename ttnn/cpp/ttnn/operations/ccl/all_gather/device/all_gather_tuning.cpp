// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// QUARANTINED TEST HOOK -- see all_gather_tuning.hpp. Delete with it.

#include "all_gather_tuning.hpp"

#include <charconv>
#include <cstdlib>
#include <string>

#include <tt_stl/assert.hpp>

namespace ttnn::operations::ccl {

namespace {

std::string_view ag_tuning_trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) {
        s.remove_prefix(1);
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) {
        s.remove_suffix(1);
    }
    return s;
}

uint32_t ag_tuning_parse_u32(std::string_view key, std::string_view value) {
    uint32_t out = 0;
    const auto* end = value.data() + value.size();
    const auto [ptr, ec] = std::from_chars(value.data(), end, out);
    TT_FATAL(ec == std::errc() && ptr == end, "TT_AG_TUNING: {}={} is not an unsigned integer", key, value);
    return out;
}

bool ag_tuning_parse_bool(std::string_view key, std::string_view value) {
    if (value == "1" || value == "true") {
        return true;
    }
    if (value == "0" || value == "false") {
        return false;
    }
    TT_THROW("TT_AG_TUNING: {}={} is not 0/1/true/false", key, value);
}

}  // namespace

Tuning Tuning::parse(std::string_view spec) {
    Tuning t;
    while (!spec.empty()) {
        const size_t comma = spec.find(',');
        const std::string_view item = ag_tuning_trim(spec.substr(0, comma));
        spec = comma == std::string_view::npos ? std::string_view{} : spec.substr(comma + 1);
        if (item.empty()) {
            continue;
        }
        const size_t eq = item.find('=');
        TT_FATAL(eq != std::string_view::npos, "TT_AG_TUNING: '{}' is not key=value", item);
        const std::string_view key = ag_tuning_trim(item.substr(0, eq));
        const std::string_view value = ag_tuning_trim(item.substr(eq + 1));

        if (key == "factory") {
            if (value == "multicast") {
                t.force_factory = AllGatherAlgorithm::Multicast;
            } else if (value == "unicast") {
                t.force_factory = AllGatherAlgorithm::Unicast;
            } else {
                TT_THROW("TT_AG_TUNING: factory={} (want multicast|unicast)", value);
            }
        } else if (key == "k") {
            if (value == "auto") {
                t.walk_k = Tuning::kAutoK;
            } else {
                t.walk_k = ag_tuning_parse_u32(key, value);
                TT_FATAL(
                    t.walk_k != 0 && (t.walk_k & (t.walk_k - 1)) == 0,
                    "TT_AG_TUNING: k={} (want auto or a power of 2)",
                    value);
            }
        } else if (key == "xfer") {
            if (value == "legacy") {
                t.xfer_policy = XferPolicy::Legacy;
            } else if (value == "segments") {
                t.xfer_policy = XferPolicy::Segments;
            } else {
                TT_THROW("TT_AG_TUNING: xfer={} (want legacy|segments)", value);
            }
        } else if (key == "rotation") {
            if (value == "none") {
                t.rotation = RotationMode::None;
            } else if (value == "absolute") {
                t.rotation = RotationMode::Absolute;
            } else if (value == "absolute_dir_stagger") {
                t.rotation = RotationMode::AbsoluteDirStagger;
            } else {
                TT_THROW("TT_AG_TUNING: rotation={} (want none|absolute|absolute_dir_stagger)", value);
            }
        } else if (key == "entry") {
            if (value == "legacy") {
                t.entry_policy = EntryPolicy::Legacy;
            } else if (value == "scatter_aware" || value == "one_packet") {
                t.entry_policy = EntryPolicy::ScatterAware;
            } else {
                TT_THROW("TT_AG_TUNING: entry={} (want legacy|scatter_aware|one_packet)", value);
            }
        } else if (key == "padded_slot") {
            t.padded_slot = ag_tuning_parse_bool(key, value);
        } else if (key == "run_cap") {
            t.run_cap_bytes = ag_tuning_parse_u32(key, value);
        } else if (key == "workers") {
            t.workers_per_dir = ag_tuning_parse_u32(key, value);
        } else if (key == "signals") {
            t.signals_per_stripe = ag_tuning_parse_u32(key, value);
        } else if (key == "mcast_cb_depth") {
            t.mcast_cb_depth = ag_tuning_parse_u32(key, value);
            // Two entries is the floor: the reader runs one entry ahead of the sender, so one deadlocks.
            TT_FATAL(*t.mcast_cb_depth >= 2, "TT_AG_TUNING: mcast_cb_depth={} (want >= 2)", value);
        } else if (key == "packet") {
            t.packet_size_override = ag_tuning_parse_u32(key, value);
        } else {
            TT_THROW(
                "TT_AG_TUNING: unknown key '{}' (known: factory, k, xfer, rotation, entry, padded_slot, run_cap, "
                "workers, signals, mcast_cb_depth, packet)",
                key);
        }
    }
    return t;
}

void Tuning::require_only_wired_keys() const {
    // Every key has a consumer as of step 2 (see APPLY2.md for where each is read), so nothing is rejected.
    // A key added later and not yet wired goes here as `TT_FATAL(field == Tuning{}.field, ...)`.
}

Tuning Tuning::from_env() {
    const char* env = std::getenv("TT_AG_TUNING");  // NOLINT(concurrency-mt-unsafe) - test hook
    return env == nullptr ? Tuning{} : parse(env);
}

}  // namespace ttnn::operations::ccl
