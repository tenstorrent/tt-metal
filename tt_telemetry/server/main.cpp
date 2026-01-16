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
#include <fstream>

#include <boost/functional/hash.hpp>
#include <cxxopts.hpp>
#include <google/protobuf/text_format.h>

#include <tt-logger/tt-logger.hpp>
#include "protobuf/factory_system_descriptor.pb.h"

#include <hal/hal.hpp>
#include <telemetry/ethernet/ethernet_helpers.hpp>
#include <telemetry/mock_telemetry_collector.hpp>
#include <telemetry/telemetry_collector.hpp>
#include <server/web_server.hpp>
#include <server/collection_endpoint.hpp>
#include <utils/hex.hpp>

/**************************************************************************************************
 Utility Functions
**************************************************************************************************/

static tt::scaleout_tools::fsd::proto::FactorySystemDescriptor load_fsd(const std::string& fsd_filename) {
    tt::scaleout_tools::fsd::proto::FactorySystemDescriptor fsd;
    std::ifstream fsd_file(fsd_filename);
    if (!fsd_file.is_open()) {
        throw std::runtime_error("Failed to open FSD file: " + fsd_filename);
    }

    std::string fsd_content((std::istreambuf_iterator<char>(fsd_file)), std::istreambuf_iterator<char>());
    fsd_file.close();

    if (!google::protobuf::TextFormat::ParseFromString(fsd_content, &fsd)) {
        throw std::runtime_error("Failed to parse FSD protobuf from file: " + fsd_filename);
    } else {
        log_info(tt::LogAlways, "Read FSD file from {}: {} bytes", fsd_filename, fsd_content.size());
    }

    return fsd;
}

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

static uint64_t get_unique_chip_id(const std::unique_ptr<tt::umd::Cluster>& cluster, tt::ChipId chip_id) {
    const std::unordered_map<tt::ChipId, uint64_t>& chip_to_unique_id =
        cluster->get_cluster_description()->get_chip_unique_ids();
    try {
        return chip_to_unique_id.at(chip_id);
    } catch (const std::out_of_range& e) {
        log_error(tt::LogAlways, "No unique ASIC ID for chip {}", chip_id);
    }
    return 0;
}

static void test_print_link_health() {
    std::cout << "Num PCIE devices: " << tt::umd::PCIDevice::enumerate_devices_info().size() << std::endl;
    std::unique_ptr<tt::umd::Cluster> cluster = std::make_unique<tt::umd::Cluster>();
    std::unique_ptr<tt::tt_metal::Hal> hal = create_hal(cluster);
    uint32_t link_up_addr =
        hal->get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::LINK_UP);

    std::cout << "Internal Connections" << std::endl << "--------------------" << std::endl;

    const std::map<tt::ChipId, std::map<tt::EthernetChannel, std::tuple<tt::ChipId, tt::EthernetChannel>>>
        ethernet_connections = get_ordered_ethernet_connections(cluster);

    for (const auto& [chip_id, remote_chip_and_channel_by_channel] : ethernet_connections) {
        // This chip...
        std::cout << "Chip " << chip_id << std::endl;

        // Iterate each channel and its remote endpoints
        for (const auto& [channel, remote_chip_and_channel] : remote_chip_and_channel_by_channel) {
            // Remote chip...
            tt::ChipId remote_chip_id;
            tt::EthernetChannel remote_channel;
            std::tie(remote_chip_id, remote_channel) = remote_chip_and_channel;

            // Print
            std::cout << "  Channel " << channel << " -> Chip " << remote_chip_id << ", Channel " << remote_channel
                      << " (Link Status: "
                      << (is_ethernet_endpoint_up(cluster, chip_id, channel, link_up_addr, false) ? "UP" : "DOWN")
                      << '/'
                      << (is_ethernet_endpoint_up(cluster, remote_chip_id, remote_channel, link_up_addr, false)
                              ? "UP"
                              : "DOWN")
                      << ')' << std::endl;
        }
    }

    // Remote off-cluster links
    std::cout << std::endl << "External Connections" << std::endl << "--------------------" << std::endl;

    const std::map<tt::ChipId, std::map<tt::EthernetChannel, std::tuple<uint64_t, tt::EthernetChannel>>>
        remote_ethernet_connections = get_ordered_ethernet_connections_to_remote_devices(cluster);

    for (const auto& [chip_id, remote_chip_and_channel_by_channel] : remote_ethernet_connections) {
        // This chip...
        std::cout << "Chip " << chip_id
                  << " [unique_id=" << fmt::format("0x{:016x}", get_unique_chip_id(cluster, chip_id)) << ']'
                  << std::endl;

        // Iterate each channel and its remote endpoints
        for (const auto& [channel, remote_chip_and_channel] : remote_chip_and_channel_by_channel) {
            // Remote chip...
            uint64_t remote_chip_unique_id;
            tt::EthernetChannel remote_channel;
            std::tie(remote_chip_unique_id, remote_channel) = remote_chip_and_channel;

            // Print
            std::cout << "  Channel " << channel << " -> [unique_id=" << fmt::format("0x{:016x}", remote_chip_unique_id)
                      << "], Channel " << remote_channel << " (Link Status: "
                      << (is_ethernet_endpoint_up(cluster, chip_id, channel, link_up_addr, false) ? "UP" : "DOWN")
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
        "fsd", "Factory system descriptor file in .textproto file", cxxopts::value<std::string>())(
        "p,port", "Port for the web server", cxxopts::value<int>()->default_value("8080"))(
        "collector-port",
        "Port for collection endpoint when in collector mode (default). Aggregators connect to this.",
        cxxopts::value<int>()->default_value("8081"))(
        "aggregate-from",
        "Comma-separated list of WebSocket endpoints to aggregate telemetry from (e.g., "
        "ws://server1:8081,ws://server2:8081). Enables aggregator mode, disabling the collection endpoint.",
        cxxopts::value<std::string>())(
        "metal-src-dir",
        "Metal source directory (optional override, auto-detected by default)",
        cxxopts::value<std::string>())("h,help", "Print usage")(
        "disable-telemetry",
        "Disables collection of telemetry. Only permitted in aggregator mode, which by default also collects local "
        "telemetry.",
        cxxopts::value<bool>()->default_value("false"))(
        "mmio-only",
        "Only collect telemetry from MMIO-capable chips (skip remote/non-MMIO chips)",
        cxxopts::value<bool>()->default_value("false"))(
        "disable-watchdog",
        "Disables the watchdog timer that monitors the telemetry thread",
        cxxopts::value<bool>()->default_value("false"))(
        "watchdog-timeout",
        "Watchdog timeout in seconds (default: 10). Watchdog will terminate the process if the telemetry thread "
        "does not advance within this time.",
        cxxopts::value<int>()->default_value("10"))(
        "failure-exposure-duration",
        "Duration in seconds to expose failure metrics before exiting when initialization fails (default: 30). "
        "This allows Prometheus time to scrape the failure state before the process exits.",
        cxxopts::value<int>()->default_value("30"));

    auto result = options.parse(argc, argv);

    if (result.contains("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    bool use_mock_telemetry = result["mock-telemetry"].as<bool>();
    bool print_link_health = result["print-link-health"].as<bool>();
    int port = result["port"].as<int>();
    int collector_port = result["collector-port"].as<int>();
    std::string metal_src_dir = "";
    if (result.contains("metal-src-dir")) {
        metal_src_dir = result["metal-src-dir"].as<std::string>();
    }
    bool telemetry_enabled = !result["disable-telemetry"].as<bool>();
    bool mmio_only = result["mmio-only"].as<bool>();

    // Watchdog configuration
    bool watchdog_disabled = result["disable-watchdog"].as<bool>();
    int watchdog_timeout = result["watchdog-timeout"].as<int>();
    if (watchdog_disabled) {
        watchdog_timeout = 0;  // 0 indicates disabled
    }

    // Failure exposure duration
    int failure_exposure_duration = result["failure-exposure-duration"].as<int>();
    if (failure_exposure_duration < 1 || failure_exposure_duration > 300) {
        log_error(
            tt::LogAlways,
            "Invalid failure-exposure-duration: {}. Must be between 1 and 300 seconds",
            failure_exposure_duration);
        return 1;
    }

    // Are we in collector (collect telemetry and export on collection endpoint) or aggregator
    // (connect to collectors and aggregate) mode?
    bool aggregator_mode = result.contains("aggregate-from");
    if (!aggregator_mode && !telemetry_enabled) {
        log_error(tt::LogAlways, "Local telemetry collection can only be disabled when in aggregator mode");
        return 1;
    }
    if (telemetry_enabled && !use_mock_telemetry && !result.contains("fsd")) {
        log_error(tt::LogAlways, "Factory system descriptor must be provided with --fsd option to collect telemetry");
        return 1;
    }
    log_info(tt::LogAlways, "Application mode: {}", aggregator_mode ? "AGGREGATOR" : "COLLECTOR");
    log_info(tt::LogAlways, "Telemetry collection: {}", telemetry_enabled ? "ENABLED" : "DISABLED");
    if (watchdog_timeout > 0) {
        log_info(tt::LogAlways, "Watchdog: ENABLED (timeout: {} seconds)", watchdog_timeout);
    } else {
        log_info(tt::LogAlways, "Watchdog: DISABLED");
    }

    // Parse aggregate-from endpoints
    std::vector<std::string> aggregate_endpoints;
    if (result.contains("aggregate-from")) {
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
        std::tie(websocket_server, websocket_subscriber) = run_collection_endpoint(collector_port);
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
        auto rtoptions = tt::llrt::RunTimeOptions();
        std::string fsd_filepath = result["fsd"].as<std::string>();
        tt::scaleout_tools::fsd::proto::FactorySystemDescriptor fsd = load_fsd(result["fsd"].as<std::string>());
        run_telemetry_collector(
            telemetry_enabled,
            subscribers,
            aggregate_endpoints,
            rtoptions,
            fsd,
            watchdog_timeout,
            failure_exposure_duration,
            mmio_only);
    }

    // Run until finished
    bool web_server_succeeded = web_server.get();
    bool websocket_succeeded = websocket_server.get();

    if (!web_server_succeeded || !websocket_succeeded) {
        return 1;
    }

    return 0;
}
