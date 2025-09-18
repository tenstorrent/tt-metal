// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * main.cpp
 * tt-telemetry main server app.
 */

 #include <algorithm>
#include <fmt/format.h>
#include <iostream>
#include <optional>
#include <sstream>

#include <boost/functional/hash.hpp>
#include <cxxopts.hpp>

#include <tt-logger/tt-logger.hpp>

#include <hal/hal.hpp>
#include <telemetry/ethernet/chip_identifier.hpp>
#include <telemetry/ethernet/ethernet_endpoint.hpp>
#include <telemetry/ethernet/ethernet_helpers.hpp>
#include <telemetry/mock_telemetry_collector.hpp>
#include <telemetry/telemetry_collector.hpp>
#include <server/web_server.hpp>
#include <server/collection_endpoint.hpp>

/**************************************************************************************************
 Utility Functions
**************************************************************************************************/

/**
 * Split a comma-separated string into a vector of trimmed strings.
 * @param input The comma-separated string to split
 * @return Vector of individual strings with whitespace trimmed
 */
std::vector<std::string> split_comma_separated(const std::string& input) {
    std::vector<std::string> result;
    std::stringstream ss(input);
    std::string item;

    while (std::getline(ss, item, ',')) {
        // Trim whitespace from both ends
        size_t start = item.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) {
            continue;  // Skip empty strings
        }
        size_t end = item.find_last_not_of(" \t\r\n");
        result.push_back(item.substr(start, end - start + 1));
    }

    return result;
}

/**************************************************************************************************
 Main
**************************************************************************************************/

static uint64_t get_unique_chip_id(const std::unique_ptr<tt::umd::Cluster>& cluster, ChipIdentifier chip_id) {
    const std::unordered_map<chip_id_t, uint64_t>& chip_to_unique_id =
        cluster->get_cluster_description()->get_chip_unique_ids();
    try {
        return chip_to_unique_id.at(chip_id.id);
    } catch (const std::out_of_range& e) {
        log_error(tt::LogAlways, "No unique ASIC ID for chip {}", chip_id);
    }
    return 0;
}

static void test_print_link_health() {
    std::cout << "Num PCIE devices: " << PCIDevice::enumerate_devices_info().size() << std::endl;
    std::unique_ptr<tt::umd::Cluster> cluster = std::make_unique<tt::umd::Cluster>();
    std::unique_ptr<tt::tt_metal::Hal> hal = create_hal(cluster);
    uint32_t link_up_addr = hal->get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::LINK_UP);

    std::cout << "Internal Connections" << std::endl << "--------------------" << std::endl;

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
            tt::umd::CoreCoord ethernet_core =
                soc_desc.get_eth_core_for_channel(channel, tt::umd::CoordSystem::LOGICAL);
            EthernetEndpoint endpoint{.chip = chip, .ethernet_core = ethernet_core, .channel = channel};

            // Remote EthernetEndpoint
            tt::umd::CoreCoord remote_ethernet_core =
                remote_soc_desc.get_eth_core_for_channel(remote_channel, tt::umd::CoordSystem::LOGICAL);
            EthernetEndpoint remote_endpoint{
                .chip = remote_chip, .ethernet_core = remote_ethernet_core, .channel = remote_channel};

            // Print
            std::cout << "  Channel " << channel << " -> [" << remote_chip << "], Channel " << remote_channel
                      << " (Link Status: " << (is_ethernet_endpoint_up(cluster, endpoint, link_up_addr) ? "UP" : "DOWN")
                      << '/' << (is_ethernet_endpoint_up(cluster, remote_endpoint, link_up_addr) ? "UP" : "DOWN") << ')'
                      << std::endl;
        }
    }

    // Remote off-cluster links
    std::cout << std::endl << "External Connections" << std::endl << "--------------------" << std::endl;

    const std::map<
        tt::umd::chip_id_t,
        std::map<tt::umd::ethernet_channel_t, std::tuple<uint64_t, tt::umd::ethernet_channel_t>>>
        remote_ethernet_connections = get_ordered_ethernet_connections_to_remote_devices(cluster);

    for (const auto& [chip_id, remote_chip_and_channel_by_channel] : remote_ethernet_connections) {
        const tt_SocDescriptor& soc_desc = cluster->get_soc_descriptor(chip_id);

        // This chip...
        tt::umd::TTDevice* device = cluster->get_tt_device(chip_id);
        ChipIdentifier chip = get_chip_identifier_from_umd_chip_id(device, chip_id);
        std::cout << chip << " [id=" << fmt::format("0x{:016x}", get_unique_chip_id(cluster, chip)) << ']' << std::endl;

        // Iterate each channel and its remote endpoints
        for (const auto& [channel, remote_chip_and_channel] : remote_chip_and_channel_by_channel) {
            // Remote chip...
            uint64_t remote_chip_unique_id;
            tt::umd::ethernet_channel_t remote_channel;
            std::tie(remote_chip_unique_id, remote_channel) = remote_chip_and_channel;

            // Local EthernetEndpoint
            tt::umd::CoreCoord ethernet_core = soc_desc.get_eth_core_for_channel(channel, tt::umd::CoordSystem::LOGICAL);
            EthernetEndpoint endpoint{.chip = chip, .ethernet_core = ethernet_core, .channel = channel};

            // Print
            std::cout << "  Channel " << channel << " -> [id=" << fmt::format("0x{:016x}", remote_chip_unique_id)
                      << "], Channel " << remote_channel
                      << " (Link Status: " << (is_ethernet_endpoint_up(cluster, endpoint, link_up_addr) ? "UP" : "DOWN")
                      << ')' << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    cxxopts::Options options("tt_telemetry_server", "TT-Metal Telemetry Server");

    options.add_options()(
        "mock-telemetry",
        "Use mock telemetry data instead of real hardware",
        cxxopts::value<bool>()->default_value("false"))(
        "print-link-health",
        "Print link health to terminal at startup",
        cxxopts::value<bool>()->default_value("false"))(
        "p,port", "Port for the web server", cxxopts::value<int>()->default_value("8080"))(
        "collector-port",
        "Port for collection endpoint when in collector mode (default). Aggregators connect to this.",
        cxxopts::value<int>()->default_value("8081"))(
        "aggregate-from",
        "Comma-separated list of WebSocket endpoints to aggregate telemetry from (e.g., "
        "ws://server1:8081,ws://server2:8081). Enables aggregator mode, disabling the collection endpoint.",
        cxxopts::value<std::string>())(
        "metal-src-dir",
        "Metal source directory (optional, defaults to TT_METAL_HOME env var)",
        cxxopts::value<std::string>())("h,help", "Print usage")(
        "disable-telemetry",
        "Disables collection of telemetry. Only permitted in aggregator mode, which by default also collects local "
        "telemetry.",
        cxxopts::value<bool>()->default_value("false"));

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    bool use_mock_telemetry = result["mock-telemetry"].as<bool>();
    bool print_link_health = result["print-link-health"].as<bool>();
    int port = result["port"].as<int>();
    int collector_port = result["collector-port"].as<int>();
    std::string metal_src_dir = "";
    if (result.count("metal-src-dir")) {
        metal_src_dir = result["metal-src-dir"].as<std::string>();
    }
    bool telemetry_enabled = !result["disable-telemetry"].as<bool>();

    // Are we in collector (collect telemetry and export on collection endpoint) or aggregator
    // (connect to collectors and aggregate) mode?
    bool aggregator_mode = result.count("aggregate-from");
    if (!aggregator_mode && !telemetry_enabled) {
        log_error(tt::LogAlways, "Local telemetry collection can only be disabled when in aggregator mode");
        return 1;
    }
    log_info(tt::LogAlways, "Application mode: {}", aggregator_mode ? "AGGREGATOR" : "COLLECTOR");
    log_info(tt::LogAlways, "Telemetry collection: {}", telemetry_enabled ? "ENABLED" : "DISABLED");

    // Parse aggregate-from endpoints
    std::vector<std::string> aggregate_endpoints;
    if (result.count("aggregate-from")) {
        std::string endpoints_str = result["aggregate-from"].as<std::string>();
        aggregate_endpoints = split_comma_separated(endpoints_str);
    }

    if (print_link_health) {
        test_print_link_health();
    }

    // Web server
    log_info(tt::LogAlways, "Starting web server on port {}", port);
    std::future<bool> web_server;
    std::shared_ptr<TelemetrySubscriber> web_server_subscriber;
    std::tie(web_server, web_server_subscriber) = run_web_server(port, metal_src_dir);

    // Subscribers are internal components that receive telemetry updates (and will export in some
    // other format)
    std::vector<std::shared_ptr<TelemetrySubscriber>> subscribers = {web_server_subscriber};

    // Collector endpoint (only in collector mode, not in aggregator mode)
    std::future<bool> websocket_server;
    std::shared_ptr<TelemetrySubscriber> websocket_subscriber;
    if (!aggregator_mode) {
        log_info(tt::LogAlways, "Starting collection endpoint on port {}", collector_port);
        std::tie(websocket_server, websocket_subscriber) = run_collection_endpoint(collector_port, metal_src_dir);
        subscribers.push_back(websocket_subscriber);
    } else {
        std::promise<bool> promise;  // create promise that immediately resolves to true
        promise.set_value(true);
        websocket_server = promise.get_future();
    }

    // Telemetry collection
    if (use_mock_telemetry) {
        // Mock telemetry
        log_info(tt::LogAlways, "Using mock telemetry data");
        MockTelemetryCollector mock_telemetry(subscribers);
        mock_telemetry.run();
    } else {
        // Real telemetry
        log_info(tt::LogAlways, "Using real hardware telemetry data");
        run_telemetry_collector(telemetry_enabled, subscribers, aggregate_endpoints);
    }

    // Run until finished
    bool web_server_succeeded = web_server.get();
    bool websocket_succeeded = websocket_server.get();

    if (!web_server_succeeded || !websocket_succeeded) {
        return 1;
    }

    return 0;
}
