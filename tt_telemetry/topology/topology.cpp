// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string_view>
#include <fmt/format.h>
#include <third_party/umd/device/api/umd/device/cluster.hpp>
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>

#include <topology/topology.hpp>

template <typename Key, typename Value, typename KeyHash>
static std::unordered_map<Value, Key> invert_map(const std::unordered_map<Key, Value, KeyHash>& map) {
    std::unordered_map<Value, Key> inverse_map;
    for (const auto& [key, value] : map) {
        inverse_map.insert({value, key});
    }
    return inverse_map;
}

// Helper: Validate host_id is within bounds
static bool is_valid_host_id(uint32_t host_id, int hosts_size) { return host_id < static_cast<uint32_t>(hosts_size); }

// Helper: Get board type for a given endpoint from FSD
static tt::BoardType get_board_type_for_endpoint(
    const tt::scaleout_tools::fsd::proto::FactorySystemDescriptor_EndPoint& endpoint,
    const tt::scaleout_tools::fsd::proto::FactorySystemDescriptor& fsd) {
    const auto& board_locations = fsd.board_types().board_locations();
    auto it = std::find_if(board_locations.begin(), board_locations.end(), [&](const auto& board_loc) {
        return board_loc.host_id() == endpoint.host_id() && board_loc.tray_id() == endpoint.tray_id();
    });

    return (it == board_locations.end()) ? tt::BoardType::UNKNOWN
                                         : tt::scaleout_tools::get_board_type_from_string(it->board_type());
}

// Helper: Get or create Board object from cache
// Separate function improves readability and enables caching
static const tt::scaleout_tools::Board& get_or_create_board(
    std::string_view hostname,
    uint32_t tray_id,
    tt::BoardType board_type,
    std::unordered_map<std::string, tt::scaleout_tools::Board>& board_cache) {
    std::string board_key = fmt::format("{}:{}", hostname, tray_id);

    // Use manual check for lazy evaluation - only create board if key doesn't exist
    auto it = board_cache.find(board_key);
    if (it == board_cache.end()) {
        it = board_cache.emplace(board_key, tt::scaleout_tools::create_board(board_type)).first;
    }
    return it->second;
}

// Helper: Build PhysicalLinkInfo for a local endpoint
// Returns nullopt if validation fails or board lookup fails
static std::optional<PhysicalLinkInfo> build_physical_link_info(
    const tt::scaleout_tools::fsd::proto::FactorySystemDescriptor_EndPoint& local_endpoint,
    const tt::scaleout_tools::fsd::proto::FactorySystemDescriptor_EndPoint& remote_endpoint,
    std::string_view remote_hostname,
    const tt::scaleout_tools::Board& board,
    const tt::scaleout_tools::fsd::proto::FactorySystemDescriptor& fsd) {
    // Get physical port for this channel
    tt::scaleout_tools::AsicChannel asic_channel{
        local_endpoint.asic_location(), tt::scaleout_tools::ChanId{local_endpoint.chan_id()}};

    try {
        const tt::scaleout_tools::Port& port = board.get_port_for_asic_channel(asic_channel);

        // Check if link is external (connected to different host)
        bool is_external = (local_endpoint.host_id() != remote_endpoint.host_id());

        // Build PhysicalLinkInfo using factory method
        if (!is_external) {
            return PhysicalLinkInfo::create(port.port_type, port.port_id);
        } else {
            if (!is_valid_host_id(remote_endpoint.host_id(), fsd.hosts_size())) {
                log_warning(tt::LogAlways, "Invalid remote host_id {} for external link", remote_endpoint.host_id());
                return std::nullopt;
            }
            const auto& remote_host = fsd.hosts()[remote_endpoint.host_id()];
            RemoteEndpointInfo remote_info{
                .hostname = std::string(remote_hostname),
                .tray = tt::tt_metal::TrayID(remote_endpoint.tray_id()),
                .asic = tt::tt_metal::ASICLocation(remote_endpoint.asic_location()),
                .channel = remote_endpoint.chan_id(),
                .aisle = remote_host.aisle(),
                .rack = remote_host.rack()};
            return PhysicalLinkInfo::create(port.port_type, port.port_id, std::move(remote_info));
        }
    } catch (const std::exception& e) {
        // Can throw if ASIC channel invalid, board config corrupt, or port mapping missing
        log_warning(
            tt::LogAlways,
            "Failed to get port for asic_channel (asic={}, channel={}): {}",
            local_endpoint.asic_location(),
            local_endpoint.chan_id(),
            e.what());
        return std::nullopt;
    }
}

// Helper: Process a single endpoint if it's on this host
static void process_endpoint_if_local(
    const tt::scaleout_tools::fsd::proto::FactorySystemDescriptor_EndPoint& endpoint,
    std::string_view endpoint_hostname,
    const tt::scaleout_tools::fsd::proto::FactorySystemDescriptor_EndPoint& remote_endpoint,
    std::string_view remote_hostname,
    std::string_view my_host_name,
    const tt::scaleout_tools::fsd::proto::FactorySystemDescriptor& fsd,
    std::unordered_map<std::string, tt::scaleout_tools::Board>& board_cache,
    std::unordered_map<EthernetEndpoint, PhysicalLinkInfo, EthernetEndpointHash>& link_info_map) {
    // Skip if not on this host
    if (endpoint_hostname != my_host_name) {
        return;
    }

    // Get board type
    tt::BoardType board_type = get_board_type_for_endpoint(endpoint, fsd);
    if (board_type == tt::BoardType::UNKNOWN) {
        log_warning(
            tt::LogAlways,
            "Could not determine board type for endpoint (host={}, tray={}, asic={}, channel={})",
            endpoint_hostname,
            endpoint.tray_id(),
            endpoint.asic_location(),
            endpoint.chan_id());
        return;
    }

    // Get or create Board object
    const auto& board = get_or_create_board(endpoint_hostname, endpoint.tray_id(), board_type, board_cache);

    // Build physical link info
    auto link_info_opt = build_physical_link_info(endpoint, remote_endpoint, remote_hostname, board, fsd);
    if (!link_info_opt.has_value()) {
        return;  // Validation or board lookup failed
    }

    // Create key and insert into map
    EthernetEndpoint eth_endpoint{
        tt::tt_metal::TrayID(endpoint.tray_id()),
        tt::tt_metal::ASICLocation(endpoint.asic_location()),
        endpoint.chan_id()};

    // Use try_emplace for efficient insertion with duplicate detection
    // inserted=false when FSD contains duplicate endpoint connections
    auto [it, inserted] = link_info_map.try_emplace(eth_endpoint, link_info_opt.value());
    if (!inserted) {
        log_warning(
            tt::LogAlways,
            "Duplicate endpoint found - skipping: (tray={}, asic={}, channel={})",
            endpoint.tray_id(),
            endpoint.asic_location(),
            endpoint.chan_id());
    }
}

// Build physical link info map by processing FSD connections
static void build_physical_link_info_map(
    std::string_view my_host_name,
    const tt::scaleout_tools::fsd::proto::FactorySystemDescriptor& fsd,
    std::unordered_map<EthernetEndpoint, PhysicalLinkInfo, EthernetEndpointHash>& endpoint_to_physical_link_info) {
    std::unordered_map<std::string, tt::scaleout_tools::Board> board_cache;

    for (const auto& connection : fsd.eth_connections().connection()) {
        const auto& endpoint_a = connection.endpoint_a();
        const auto& endpoint_b = connection.endpoint_b();

        // Validate host IDs using helper
        if (!is_valid_host_id(endpoint_a.host_id(), fsd.hosts_size()) ||
            !is_valid_host_id(endpoint_b.host_id(), fsd.hosts_size())) {
            log_warning(
                tt::LogAlways,
                "Skipping connection with invalid host_id (host_a={}, host_b={})",
                endpoint_a.host_id(),
                endpoint_b.host_id());
            continue;
        }

        const std::string& hostname_a = fsd.hosts()[endpoint_a.host_id()].hostname();
        const std::string& hostname_b = fsd.hosts()[endpoint_b.host_id()].hostname();

        // Process endpoint A if it's on this host
        process_endpoint_if_local(
            endpoint_a,
            hostname_a,
            endpoint_b,
            hostname_b,
            my_host_name,
            fsd,
            board_cache,
            endpoint_to_physical_link_info);

        // Process endpoint B if it's on this host
        // Both endpoints local for intra-host links (e.g., chip-to-chip on same host)
        process_endpoint_if_local(
            endpoint_b,
            hostname_b,
            endpoint_a,
            hostname_a,
            my_host_name,
            fsd,
            board_cache,
            endpoint_to_physical_link_info);
    }

    log_info(
        tt::LogAlways,
        "TopologyHelper: Built physical link info map with {} endpoints",
        endpoint_to_physical_link_info.size());
}

TopologyHelper::TopologyHelper(
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const std::unique_ptr<tt::tt_metal::PhysicalSystemDescriptor>& psd,
    const tt::scaleout_tools::fsd::proto::FactorySystemDescriptor& fsd) :
    my_host_name(psd->my_host_name()) {
    // Get mapping of chip ID <-> ASIC unique ID. This is only valid for the local host!
    const std::unordered_map<tt::ChipId, uint64_t>& chip_id_to_unique_id =
        cluster->get_cluster_description()->get_chip_unique_ids();
    std::unordered_map<uint64_t, tt::ChipId> unique_id_to_chip_id = invert_map(chip_id_to_unique_id);

    // Produce the following maps (all valid only for the local host because chip ID is not
    // globally unique):
    //  (asic_location,tray_id) -> chip_id
    //  chip_id -> (asic_location,tray_id)
    //  chip_id -> ASICDescriptor
    for (auto [unique_id, asic_descriptor] : psd->get_asic_descriptors()) {
        auto it = unique_id_to_chip_id.find(*unique_id);
        if (it == unique_id_to_chip_id.end()) {
            continue;
        }
        tt::ChipId chip_id = it->second;

        auto key = std::make_pair(asic_descriptor.asic_location, asic_descriptor.tray_id);
        TT_FATAL(
            asic_location_and_tray_id_to_local_chip_id_.count(key) == 0,
            "Duplicate key (asic_location={}, tray_id={}) found in mapping",
            *asic_descriptor.asic_location,
            *asic_descriptor.tray_id);
        asic_location_and_tray_id_to_local_chip_id_.insert({key, chip_id});
        local_chip_id_to_asic_descriptor_.insert({chip_id, asic_descriptor});
    }

    // Build physical link info map from FSD
    build_physical_link_info_map(my_host_name, fsd, endpoint_to_physical_link_info_);
}

std::optional<tt::ChipId> TopologyHelper::get_local_chip_id_for_asic_location_and_tray(
    tt::tt_metal::ASICLocation asic_location, tt::tt_metal::TrayID tray_id) {
    auto it = asic_location_and_tray_id_to_local_chip_id_.find({asic_location, tray_id});
    if (it == asic_location_and_tray_id_to_local_chip_id_.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::optional<tt::tt_metal::ASICDescriptor> TopologyHelper::get_asic_descriptor_for_local_chip(tt::ChipId chip_id) {
    auto it = local_chip_id_to_asic_descriptor_.find(chip_id);
    if (it == local_chip_id_to_asic_descriptor_.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::optional<PhysicalLinkInfo> TopologyHelper::get_physical_link_info(const EthernetEndpoint& endpoint) const {
    if (auto it = endpoint_to_physical_link_info_.find(endpoint); it != endpoint_to_physical_link_info_.end()) {
        return it->second;
    }
    return std::nullopt;
}
