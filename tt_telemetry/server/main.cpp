/*
 * main.cpp
 * tt-telemetry main server app.
 */

 #include <algorithm>
 #include <iostream>
 #include <optional>

#include <boost/functional/hash.hpp>
#include <cxxopts.hpp>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/assert.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include "impl/context/metal_context.hpp"

#include <telemetry/ethernet/chip_identifier.hpp>
#include <telemetry/ethernet/ethernet_endpoint.hpp>
#include <telemetry/ethernet/ethernet_link.hpp>
#include <telemetry/ethernet/ethernet_helpers.hpp>
#include <telemetry/ethernet/print_helpers.hpp>
#include <telemetry/mock_telemetry_provider.hpp>
#include <telemetry/telemetry_provider.hpp>
#include <server/web_server.hpp>

// Function prototype for sysfs metrics discovery
void print_sysfs_metrics();
void test_umd();

/**************************************************************************************************
 Main
**************************************************************************************************/

// Note: this is apparently a cached link status and not live!
static bool is_ethernet_endpoint_up(
    const tt::Cluster& cluster, tt::umd::chip_id_t chip_id, tt::umd::ethernet_channel_t ethernet_channel) {
    for (const auto& [core, channel] : cluster.get_soc_desc(chip_id).logical_eth_core_to_chan_map) {
        if (ethernet_channel == channel) {
            // Found the channel on the chip we are interested in, we now have the core coordinates
            return cluster.is_ethernet_link_up(chip_id, core);
        }
    }

    // Invalid chip ID or channel -> not connected
    return false;
}

static void old_test_print_link_health() {
    const tt::tt_metal::MetalContext& instance = tt::tt_metal::MetalContext::instance();
    const tt::Cluster& cluster = instance.get_cluster();
    const std::map<
        tt::umd::chip_id_t,
        std::map<tt::umd::ethernet_channel_t, std::tuple<tt::umd::chip_id_t, tt::umd::ethernet_channel_t>>>
        ethernet_connections = get_ordered_ethernet_connections(cluster);
    tt::tt_metal::ClusterType cluster_type = cluster.get_cluster_type();

    std::cout << "Cluster Type: " << cluster_type << std::endl;

    for (const auto& [chip_id, remote_chip_and_channel_by_channel] : ethernet_connections) {
        std::cout << get_chip_identifier_from_umd_chip_id(cluster, chip_id) << std::endl;
        for (const auto& [channel, remote_chip_and_channel] : remote_chip_and_channel_by_channel) {
            tt::umd::chip_id_t remote_chip_id;
            tt::umd::ethernet_channel_t remote_channel;
            std::tie(remote_chip_id, remote_channel) = remote_chip_and_channel;
            std::cout << "  Channel " << channel << " -> ["
                      << get_chip_identifier_from_umd_chip_id(cluster, remote_chip_id) << "], Channel "
                      << remote_channel
                      << " (Link Status: " << (is_ethernet_endpoint_up(cluster, chip_id, channel) ? "UP" : "DOWN")
                      << '/' << (is_ethernet_endpoint_up(cluster, remote_chip_id, remote_channel) ? "UP" : "DOWN")
                      << ')' << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "Links:" << std::endl;

    std::vector<EthernetLink> links = get_ethernet_links(cluster);

    for (const auto& link : links) {
        std::cout << link.first << " <--> " << link.second << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Endpoints:" << std::endl;

    for (const auto& [chip_id, endpoints] : get_ethernet_endpoints_by_chip(cluster)) {
        std::cout << chip_id << std::endl;
        for (const auto& endpoint : endpoints) {
            std::cout << "  " << endpoint << ": " << (is_ethernet_endpoint_up(cluster, endpoint) ? "UP" : "DOWN")
                      << std::endl;
        }
    }
}

static void test_print_link_health() {
    std::cout << "Num PCIE devices: " << PCIDevice::enumerate_devices_info().size() << std::endl;
    std::unique_ptr<tt::umd::Cluster> cluster = std::make_unique<tt::umd::Cluster>();

    const std::map<
        tt::umd::chip_id_t,
        std::map<tt::umd::ethernet_channel_t, std::tuple<tt::umd::chip_id_t, tt::umd::ethernet_channel_t>>>
        ethernet_connections = get_ordered_ethernet_connections(cluster);

    for (const auto& [chip_id, remote_chip_and_channel_by_channel] : ethernet_connections) {
        // Create a SOC descriptor just for the purpose of mapping Ethernet channel to core coordinates
        const tt_SocDescriptor& soc_desc = cluster->get_soc_descriptor(chip_id);

        // This chip...
        tt::umd::TTDevice* device = cluster->get_tt_device(chip_id);
        ChipIdentifier chip = get_chip_identifier_from_umd_chip_id(device, chip_id);
        std::cout << chip << std::endl;

        // Iterate each channel and its remote endpoints
        for (const auto& [channel, remote_chip_and_channel] : remote_chip_and_channel_by_channel) {
            // Remote chip...
            tt::umd::chip_id_t remote_chip_id;
            tt::umd::ethernet_channel_t remote_channel;
            std::tie(remote_chip_id, remote_channel) = remote_chip_and_channel;
            tt::umd::TTDevice* remote_device = cluster->get_tt_device(remote_chip_id);
            const tt_SocDescriptor& remote_soc_desc = cluster->get_soc_descriptor(remote_chip_id);
            ChipIdentifier remote_chip = get_chip_identifier_from_umd_chip_id(remote_device, remote_chip_id);

            // Local EthernetEndpoint
            CoreCoord ethernet_core = soc_desc.get_eth_core_for_channel(channel, tt::umd::CoordSystem::LOGICAL);
            EthernetEndpoint endpoint{.chip = chip, .ethernet_core = ethernet_core, .channel = channel};

            // Remote EthernetEndpoint
            CoreCoord remote_ethernet_core =
                remote_soc_desc.get_eth_core_for_channel(remote_channel, tt::umd::CoordSystem::LOGICAL);
            EthernetEndpoint remote_endpoint{
                .chip = remote_chip, .ethernet_core = remote_ethernet_core, .channel = remote_channel};

            // Print
            std::cout << "  Channel " << channel << " -> [" << remote_chip << "], Channel " << remote_channel
                      << " (Link Status: " << (is_ethernet_endpoint_up(cluster, endpoint) ? "UP" : "DOWN") << '/'
                      << (is_ethernet_endpoint_up(cluster, remote_endpoint) ? "UP" : "DOWN") << ')' << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    // std::cout << "crc_err_addr = " << std::hex
    //           << tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
    //                  tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::CRC_ERR)
    //           << std::endl;
    // std::cout << "retrain_count_addr = "
    //           << tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
    //                  tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::RETRAIN_COUNT)
    //           << std::endl;
    // std::cout << "corr_cw_addr = "
    //           << tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
    //                  tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::CORR_CW)
    //           << std::endl;
    // std::cout << "uncorr_cw_addr = "
    //           << tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
    //                  tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNCORR_CW)
    //           << std::endl;
    // return 0;

    // Parse command line arguments
    cxxopts::Options options("tt_telemetry_server", "TT-Metal Telemetry Server");

    options.add_options()(
        "mock-telemetry",
        "Use mock telemetry data instead of real hardware",
        cxxopts::value<bool>()->default_value("false"))(
        "print-link-health",
        "Print link health to terminal at startup",
        cxxopts::value<bool>()->default_value("false"))(
        "p,port", "Port for the primary web server", cxxopts::value<int>()->default_value("8080"))(
        "h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    bool use_mock_telemetry = result["mock-telemetry"].as<bool>();
    bool print_link_health = result["print-link-health"].as<bool>();
    int port = result["port"].as<int>();

    if (print_link_health) {
        test_print_link_health();
    }

    // Web server
    std::cout << "Starting primary web server on port " << port << std::endl;
    std::future<bool> web_server;
    std::shared_ptr<TelemetrySubscriber> web_server_subscriber;
    std::tie(web_server, web_server_subscriber) = run_web_server(port);

    // Web server #2 (testing the ability of our producer to supply two consumers)
    std::cout << "Starting secondary web server on port 5555" << std::endl;
    std::future<bool> web_server2;
    std::shared_ptr<TelemetrySubscriber> web_server2_subscriber;
    std::tie(web_server2, web_server2_subscriber) = run_web_server(5555);

    if (use_mock_telemetry) {
        // Mock telemetry
        std::cout << "Using mock telemetry data" << std::endl;
        MockTelemetryProvider mock_telemetry{web_server_subscriber, web_server2_subscriber};
        mock_telemetry.run();
    } else {
        // Real telemetry
        std::cout << "Using real hardware telemetry data" << std::endl;
        run_telemetry_provider({web_server_subscriber, web_server2_subscriber});
    }

    // Run until finished
    bool web_server_succeeded = web_server.get();
    bool web_server2_succeeded = web_server2.get();
    if (!web_server_succeeded || !web_server2_succeeded) {
        return 1;
    }

    return 0;
}
