#include <boost/functional/hash.hpp>

#include <tt-metalium/control_plane.hpp>

#include <telemetry/ethernet/chip_identifier.hpp>

bool GalaxyUbbIdentifier::operator<(const GalaxyUbbIdentifier &other) const {
    if (tray_id != other.tray_id) {
        return tray_id < other.tray_id;
    }
    return asic_id < other.asic_id;
}

bool GalaxyUbbIdentifier::operator==(const GalaxyUbbIdentifier &other) const {
    return tray_id == other.tray_id && asic_id == other.asic_id;
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
        auto asic_id = galaxy_ubb.value().asic_id;
        return { "tray" + std::to_string(tray_id), "n" + std::to_string(asic_id) };
    }
    return { "chip" + std::to_string(id) };
}

std::ostream &operator<<(std::ostream &os, const ChipIdentifier &chip) {
    if (chip.galaxy_ubb.has_value()) {
        os << "Tray " << chip.galaxy_ubb.value().tray_id << ", N" << chip.galaxy_ubb.value().asic_id << " (Chip " << chip.id << ')';
    } else {
        os << "Chip " << chip.id;
    }
    return os;
}

// Required by Boost for hashing
size_t hash_value(const GalaxyUbbIdentifier &g) {
    size_t seed = 0;
    boost::hash_combine(seed, g.tray_id);
    boost::hash_combine(seed, g.asic_id);
    return seed;
}

size_t hash_value(const ChipIdentifier &c) {
    size_t seed = 0;
    boost::hash_combine(seed, c.id);
    boost::hash_combine(seed, c.galaxy_ubb);
    return seed;
}

ChipIdentifier get_chip_identifier_from_umd_chip_id(const tt::Cluster &cluster, chip_id_t chip_id) {
    if (cluster.get_cluster_type() == tt::tt_metal::ClusterType::GALAXY) {
        const std::unordered_map<tt::ARCH, std::vector<std::uint16_t>> ubb_bus_ids = {
            {tt::ARCH::WORMHOLE_B0, {0xC0, 0x80, 0x00, 0x40}},
            {tt::ARCH::BLACKHOLE, {0x00, 0x40, 0xC0, 0x80}},
        };
        const auto& tray_bus_ids = ubb_bus_ids.at(cluster.arch());
        const auto bus_id = cluster.get_bus_id(chip_id);
        auto tray_bus_id_it = std::find(tray_bus_ids.begin(), tray_bus_ids.end(), bus_id & 0xF0);
        if (tray_bus_id_it != tray_bus_ids.end()) {
            auto ubb_asic_id = bus_id & 0x0F;
            return { .id = chip_id, .galaxy_ubb = GalaxyUbbIdentifier{tray_bus_id_it - tray_bus_ids.begin() + 1, ubb_asic_id} };
        }

        // Invalid UBB, drop through
    }

    // Not a known cluster type, just use chip ID directly
    return { .id = chip_id, .galaxy_ubb = {} }; // invalid UBB ID if not found
}