/*
 * main.cpp
 * tt-telemetry main server app.
 * 
 * TODO
 * ----
 * - Add the logic that test_system_health.cpp uses to determine number of connections and write
 *   a loop that iterates them all and prints health.
 * - cluster.get_ethernet_connections() <-- Does this return chips on other hosts in a cluster?
 *   What about the remote side of each connection? Need to test on a multi-host system.
 * - Need to handle other cluster types (including N300, etc., which have most of their Ethernet
 *   cores unused), ensuring we don't mark legitimately unused connections as "down".
 * - Simple REST exporter and web GUI.  
 */

#include <algorithm>
#include <iostream>
#include <optional>

#include <boost/functional/hash.hpp>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/assert.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include "impl/context/metal_context.hpp"

#include <telemetry/chip_identifier.hpp>
#include <telemetry/ethernet_endpoint.hpp>
#include <telemetry/ethernet_link.hpp>
#include <telemetry/ethernet_helpers.hpp>
#include <telemetry/print_helpers.hpp>


int main() {
    const tt::tt_metal::MetalContext &instance = tt::tt_metal::MetalContext::instance();
    const tt::Cluster &cluster = instance.get_cluster();
    const std::map<
        tt::umd::chip_id_t, 
        std::map<
            tt::umd::ethernet_channel_t, 
            std::tuple<tt::umd::chip_id_t, tt::umd::ethernet_channel_t>
        >
    > ethernet_connections = get_ordered_ethernet_connections(cluster);
    tt::tt_metal::ClusterType cluster_type = cluster.get_cluster_type();

    std::cout << "Cluster Type: " << cluster_type << std::endl;

    for (const auto &[chip_id, remote_chip_and_channel_by_channel]: ethernet_connections) {
        std::cout << get_chip_identifier_from_umd_chip_id(cluster, chip_id) << std::endl;
        for (const auto &[channel, remote_chip_and_channel]: remote_chip_and_channel_by_channel) {
            tt::umd::chip_id_t remote_chip_id;
            tt::umd::ethernet_channel_t remote_channel;
            std::tie(remote_chip_id, remote_channel) = remote_chip_and_channel;
            std::cout << "  Channel " << channel << " -> [" << get_chip_identifier_from_umd_chip_id(cluster, remote_chip_id) 
                      << "], Channel " << remote_channel 
                      << " (Link Status: " << (is_ethernet_endpoint_up(cluster, chip_id, channel) ? "UP" : "DOWN") << '/' << (is_ethernet_endpoint_up(cluster, remote_chip_id, remote_channel) ? "UP" : "DOWN") <<  ')'
                      << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "Links:" << std::endl;

    std::vector<EthernetLink> links = get_ethernet_links(cluster);

    for (const auto &link: links) {
        std::cout << link.first << " <--> " << link.second << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Endpoints:" << std::endl;

    for (const auto &[chip_id, endpoints]: get_ethernet_endpoints_by_chip(cluster)) {
        std::cout << chip_id << std::endl;
        for (const auto &endpoint: endpoints) {
            std::cout << "  " << endpoint << ": " << (is_ethernet_endpoint_up(cluster, endpoint) ? "UP" : "DOWN") << std::endl;
        }
    }

    return 0;
}