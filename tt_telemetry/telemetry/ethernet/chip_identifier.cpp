// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/functional/hash.hpp>

#include <tt-metalium/control_plane.hpp>

#include <telemetry/ethernet/chip_identifier.hpp>

bool GalaxyUbbIdentifier::operator<(const GalaxyUbbIdentifier &other) const {
    if (tray_id != other.tray_id) {
        return tray_id < other.tray_id;
    }
    return chip_number < other.chip_number;
}

bool GalaxyUbbIdentifier::operator==(const GalaxyUbbIdentifier &other) const {
    return tray_id == other.tray_id && chip_number == other.chip_number;
}

bool ChipIdentifier::operator<(const ChipIdentifier &other) const {
    if (galaxy_ubb != other.galaxy_ubb) {
        return galaxy_ubb < other.galaxy_ubb;
    }
    return id < other.id;
}

bool ChipIdentifier::operator==(const ChipIdentifier &other) const {
    return id == other.id && galaxy_ubb == other.galaxy_ubb;
}

std::vector<std::string> ChipIdentifier::telemetry_path() const {
    if (galaxy_ubb.has_value()) {
        auto tray_id = galaxy_ubb.value().tray_id;
        auto chip_number = galaxy_ubb.value().chip_number;
        return { "tray" + std::to_string(tray_id), "n" + std::to_string(chip_number) };
    }
    return { "chip" + std::to_string(id) };
}

std::ostream &operator<<(std::ostream &os, const ChipIdentifier &chip) {
    if (chip.galaxy_ubb.has_value()) {
        os << "Tray " << chip.galaxy_ubb.value().tray_id << ", N" << chip.galaxy_ubb.value().chip_number << " (Chip " << chip.id << ')';
    } else {
        os << "Chip " << chip.id;
    }
    return os;
}

// Required by Boost for hashing
size_t hash_value(const GalaxyUbbIdentifier &g) {
    size_t seed = 0;
    boost::hash_combine(seed, g.tray_id);
    boost::hash_combine(seed, g.chip_number);
    return seed;
}

size_t hash_value(const ChipIdentifier &c) {
    size_t seed = 0;
    boost::hash_combine(seed, c.id);
    boost::hash_combine(seed, c.galaxy_ubb);
    return seed;
}

static uint16_t get_bus_id(tt::umd::TTDevice* device) { return device->get_pci_device()->get_device_info().pci_bus; }

ChipIdentifier get_chip_identifier_from_umd_chip_id(tt::umd::TTDevice* device, tt::umd::chip_id_t chip_id) {
    if (device->get_board_type() == BoardType::UBB) {
        // UBB is the Galaxy 6U board type
        const std::unordered_map<tt::ARCH, std::vector<std::uint16_t>> ubb_bus_ids = {
            {tt::ARCH::WORMHOLE_B0, {0xC0, 0x80, 0x00, 0x40}},
            {tt::ARCH::BLACKHOLE, {0x00, 0x40, 0xC0, 0x80}},
        };
        const auto& tray_bus_ids = ubb_bus_ids.at(device->get_arch());
        const auto bus_id = get_bus_id(device);
        auto tray_bus_id_it = std::find(tray_bus_ids.begin(), tray_bus_ids.end(), bus_id & 0xF0);
        if (tray_bus_id_it != tray_bus_ids.end()) {
            auto ubb_chip_number = bus_id & 0x0F;
            return {
                .id = chip_id,
                .galaxy_ubb = GalaxyUbbIdentifier{tray_bus_id_it - tray_bus_ids.begin() + 1, ubb_chip_number}};
        }

        // Invalid UBB, drop through
    }

    // Not a known cluster type, just use chip ID directly
    return {.id = chip_id, .galaxy_ubb = {}};  // invalid UBB ID if not found
}
