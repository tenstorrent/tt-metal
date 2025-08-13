#include <telemetry/ethernet/print_helpers.hpp>

std::ostream &operator<<(std::ostream &os, const tt::tt_metal::ClusterType cluster_type) {
    switch (cluster_type) {
    case tt::tt_metal::ClusterType::INVALID:    os << "Invalid"; break;
    case tt::tt_metal::ClusterType::N150:       os << "N150"; break;
    case tt::tt_metal::ClusterType::N300:       os << "N300"; break;
    case tt::tt_metal::ClusterType::T3K:        os << "T3K"; break;
    case tt::tt_metal::ClusterType::GALAXY:     os << "Galaxy"; break;
    case tt::tt_metal::ClusterType::TG:         os << "TG"; break;
    case tt::tt_metal::ClusterType::P100:       os << "P100"; break;
    case tt::tt_metal::ClusterType::P150:       os << "P150"; break;
    case tt::tt_metal::ClusterType::P150_X2:    os << "P150 x2"; break;
    case tt::tt_metal::ClusterType::P150_X4:    os << "P150 x4"; break;
    case tt::tt_metal::ClusterType::SIMULATOR_WORMHOLE_B0: os << "Simulator Blackhole B0"; break;
    case tt::tt_metal::ClusterType::SIMULATOR_BLACKHOLE:   os << "Simulator Blackhole"; break;
    case tt::tt_metal::ClusterType::N300_2x2:   os << "N300 2x2"; break;
    default:
        os << "Unknown (" << int(cluster_type) << ')';
        break;
    }
    return os;
}