// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/*
 * topology/topology.hpp
 *
 * Utilities for understanding topology and mapping between different identifiers.
 */

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <boost/functional/hash.hpp>

#include <tt_metal/fabric/physical_system_descriptor.hpp>
#include <protobuf/factory_system_descriptor.pb.h>
#include <board/board.hpp>

namespace tt::umd {
class Cluster;
}

// Hash function for std::pair<tt::tt_metal::ASICLocation, tt::tt_metal::TrayID>
struct ASICLocationAndTrayIDHash {
    std::size_t operator()(const std::pair<tt::tt_metal::ASICLocation, tt::tt_metal::TrayID>& p) const noexcept {
        std::size_t seed = 0;
        boost::hash_combine(seed, *p.first);
        boost::hash_combine(seed, *p.second);
        return seed;
    }
};

/*
 * EthernetEndpoint
 *
 * Uniquely identifies an Ethernet endpoint by tray, ASIC location, and channel.
 * Hostname is implicit (TopologyHelper only tracks local endpoints).
 * ChipId is excluded because it changes between runs.
 */
struct EthernetEndpoint {
    tt::tt_metal::TrayID tray_id{};
    tt::tt_metal::ASICLocation asic_location{};
    uint32_t channel{0};

    bool operator==(const EthernetEndpoint& other) const noexcept = default;
};

// Hash function for EthernetEndpoint to use as unordered_map key
struct EthernetEndpointHash {
    std::size_t operator()(const EthernetEndpoint& ep) const noexcept {
        std::size_t seed = 0;
        boost::hash_combine(seed, *ep.tray_id);
        boost::hash_combine(seed, *ep.asic_location);
        boost::hash_combine(seed, ep.channel);
        return seed;
    }
};

/*
 * RemoteEndpointInfo
 *
 * Information about the remote endpoint of an external Ethernet link.
 * Includes both logical endpoint identification and physical datacenter location.
 */
struct RemoteEndpointInfo {
    std::string hostname{};
    tt::tt_metal::TrayID tray{};
    tt::tt_metal::ASICLocation asic{};
    uint32_t channel{0};
    std::string aisle{};
    uint32_t rack{0};

    bool operator==(const RemoteEndpointInfo& other) const noexcept = default;
};

/*
 * PhysicalLinkInfo
 *
 * Physical topology information for an Ethernet link endpoint.
 * Contains immutable properties that identify the physical port and remote connection.
 *
 * Use factory method to create instances. Link is external iff remote_endpoint is present.
 */
struct PhysicalLinkInfo {
    // Physical port on this board
    tt::scaleout_tools::PortType port_type{};
    tt::scaleout_tools::PortId port_id{};

    // Remote endpoint information (only present for external links)
    std::optional<RemoteEndpointInfo> remote_endpoint{};

    // Factory method to create PhysicalLinkInfo
    // For internal links: omit remote_endpoint (uses default nullopt)
    // For external links: provide remote_endpoint info
    static PhysicalLinkInfo create(
        tt::scaleout_tools::PortType port_type,
        tt::scaleout_tools::PortId port_id,
        std::optional<RemoteEndpointInfo> remote_endpoint = std::nullopt) noexcept {
        return PhysicalLinkInfo{port_type, port_id, std::move(remote_endpoint)};
    }

    // Query whether this link is external (connected to different host)
    bool is_external() const noexcept { return remote_endpoint.has_value(); }

    bool operator==(const PhysicalLinkInfo& other) const noexcept = default;
};

// TODO: Open to a better name for this
class TopologyHelper {
public:
    TopologyHelper(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        const std::unique_ptr<tt::tt_metal::PhysicalSystemDescriptor>& psd,
        const tt::scaleout_tools::fsd::proto::FactorySystemDescriptor& fsd);
    std::optional<tt::ChipId> get_local_chip_id_for_asic_location_and_tray(
        tt::tt_metal::ASICLocation asic_location, tt::tt_metal::TrayID tray_id);
    std::optional<tt::tt_metal::ASICDescriptor> get_asic_descriptor_for_local_chip(tt::ChipId chip_id);
    std::optional<PhysicalLinkInfo> get_physical_link_info(const EthernetEndpoint& endpoint) const;

    const std::string my_host_name;

private:
    std::unordered_map<
        std::pair<tt::tt_metal::ASICLocation, tt::tt_metal::TrayID>,
        tt::ChipId,
        ASICLocationAndTrayIDHash>
        asic_location_and_tray_id_to_local_chip_id_;
    std::unordered_map<tt::ChipId, tt::tt_metal::ASICDescriptor> local_chip_id_to_asic_descriptor_;
    std::unordered_map<EthernetEndpoint, PhysicalLinkInfo, EthernetEndpointHash> endpoint_to_physical_link_info_;
};
