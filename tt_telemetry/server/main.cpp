// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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

#include <hal/hal.hpp>
#include <telemetry/ethernet/chip_identifier.hpp>
#include <telemetry/ethernet/ethernet_endpoint.hpp>
#include <telemetry/ethernet/ethernet_helpers.hpp>
#include <telemetry/mock_telemetry_provider.hpp>
#include <telemetry/telemetry_provider.hpp>
#include <server/web_server.hpp>


/**************************************************************************************************
 Main
**************************************************************************************************/

static void test_print_link_health() {
    std::cout << "Num PCIE devices: " << PCIDevice::enumerate_devices_info().size() << std::endl;
    std::unique_ptr<tt::umd::Cluster> cluster = std::make_unique<tt::umd::Cluster>();
    std::unique_ptr<tt::tt_metal::Hal> hal = create_hal(cluster);
    uint32_t link_up_addr = hal->get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::LINK_UP);

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
            tt::umd::CoreCoord ethernet_core = soc_desc.get_eth_core_for_channel(channel, tt::umd::CoordSystem::LOGICAL);
            EthernetEndpoint endpoint{.chip = chip, .ethernet_core = ethernet_core, .channel = channel};

            // Remote EthernetEndpoint
            tt::umd::CoreCoord remote_ethernet_core =
                remote_soc_desc.get_eth_core_for_channel(remote_channel, tt::umd::CoordSystem::LOGICAL);
            EthernetEndpoint remote_endpoint{
                .chip = remote_chip, .ethernet_core = remote_ethernet_core, .channel = remote_channel};

            // Print
            std::cout << "  Channel " << channel << " -> [" << remote_chip << "], Channel " << remote_channel
                      << " (Link Status: " << (is_ethernet_endpoint_up(cluster, endpoint, link_up_addr) ? "UP" : "DOWN") << '/'
                      << (is_ethernet_endpoint_up(cluster, remote_endpoint, link_up_addr) ? "UP" : "DOWN") << ')' << std::endl;
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
    log_info(tt::LogAlways, "Starting primary web server on port {}", port);
    std::future<bool> web_server;
    std::shared_ptr<TelemetrySubscriber> web_server_subscriber;
    std::tie(web_server, web_server_subscriber) = run_web_server(port);

    // Web server #2 (testing the ability of our producer to supply two consumers)
    uint16_t secondary_port = 5555;
    log_info(tt::LogAlways, "Starting secondary web server on port {}", secondary_port);
    std::future<bool> web_server2;
    std::shared_ptr<TelemetrySubscriber> web_server2_subscriber;
    std::tie(web_server2, web_server2_subscriber) = run_web_server(secondary_port);

    if (use_mock_telemetry) {
        // Mock telemetry
        log_info(tt::LogAlways, "Using mock telemetry data");
        MockTelemetryProvider mock_telemetry{web_server_subscriber, web_server2_subscriber};
        mock_telemetry.run();
    } else {
        // Real telemetry
        log_info(tt::LogAlways, "Using real hardware telemetry data");
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
