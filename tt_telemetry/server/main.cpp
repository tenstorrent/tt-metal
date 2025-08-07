/*
 * main.cpp
 * tt-telemetry main server app.
 * 
 * Notes
 * -----
 * - Ethernet connections returned by tt::Cluster::get_ethernet_connections() are discovered in
 *   TopologyDiscovery::discover_remote_chips() in topology_discovery.cpp. This function seems to
 *   exclude connections that are not active. So if a link is inactive at the time topology 
 *   discovery runs, it will never be known. test_system_health.cpp checks that an expected number
 *   of active connections per chip are present (without identifying their specific channel). We
 *   will eventually have to use a system descriptor file to discover required connections rather
 *   than just calling get_ethernet_connections(). ScaleoutTopologyManager at this time is
 *   proposing a cabling spec, which should make it possible to construct a set of intended
 *   external connections. Internal (chip to chip) connections should be possible to infer from the
 *   system type and number of chips found on the host.
 * - For now, lacking a mechanism to obtain a list of desired connections, we simply report on the
 *   discovered ones to proceed with a telemtry PoC. Eventually, this must be fixed.
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
#include <server/web_server.hpp>
#include <server/json_messages.hpp>


/**************************************************************************************************
 Main
**************************************************************************************************/

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

    // JSON test
    messages::EndpointDefinitionMessage msg {
        .host = "sjc-wh-05",
        .endpoints = { { 200, "from", "to" } }
    };

    nlohmann::json j = msg;
    std::cout << "json: " << j << std::endl;;

    messages::EndpointStateChangeMessage msg2 {
        .host = "sjc-wh-05",
        .endpoints = { { 100, true } }
    };

    j = msg2;
    std::cout << "json: " << j << std::endl;

    // Web server
    run_web_server();

    return 0;
}