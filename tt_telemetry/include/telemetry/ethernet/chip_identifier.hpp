#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/ethernet/chip_identifier.hpp
 *
 * Enriched chip identification, including information about tray and board position, where
 * applicable (e.g., on Galaxy machines). The basic tt::umd::chip_id_t is always the fallback.
 */

#include <ostream>
#include <fmt/format.h>

#include <third_party/umd/device/api/umd/device/cluster.h>
#include <third_party/umd/device/api/umd/device/types/cluster_descriptor_types.h>

struct GalaxyUbbIdentifier {
    uint32_t tray_id;
    uint32_t chip_number;

    bool operator<(const GalaxyUbbIdentifier &other) const;
    bool operator==(const GalaxyUbbIdentifier &other) const;
};

struct ChipIdentifier {
    tt::umd::chip_id_t id;
    std::optional<GalaxyUbbIdentifier> galaxy_ubb;

    bool operator<(const ChipIdentifier &other) const;
    bool operator==(const ChipIdentifier &other) const;

    std::vector<std::string> telemetry_path() const;
};

std::ostream &operator<<(std::ostream &os, const ChipIdentifier &chip);

size_t hash_value(const GalaxyUbbIdentifier &g);
size_t hash_value(const ChipIdentifier &c);

namespace std {
    template<>
    struct hash<GalaxyUbbIdentifier> {
        size_t operator()(const GalaxyUbbIdentifier &g) const noexcept {
            return hash_value(g);
        }
    };

    template<>
    struct hash<ChipIdentifier> {
        size_t operator()(const ChipIdentifier &c) const noexcept {
            return hash_value(c);
        }
    };
}

ChipIdentifier get_chip_identifier_from_umd_chip_id(tt::umd::TTDevice* device, tt::umd::chip_id_t chip_id);

// fmt formatter for ChipIdentifier
template <>
struct fmt::formatter<ChipIdentifier> {
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.end();
    }

    template <typename FormatContext>
    auto format(const ChipIdentifier& chip, FormatContext& ctx) const -> decltype(ctx.out()) {
        if (chip.galaxy_ubb.has_value()) {
            return fmt::format_to(
                ctx.out(),
                "Tray {}, N{} (Chip {})",
                chip.galaxy_ubb.value().tray_id,
                chip.galaxy_ubb.value().chip_number,
                chip.id);
        } else {
            return fmt::format_to(ctx.out(), "Chip {}", chip.id);
        }
    }
};
