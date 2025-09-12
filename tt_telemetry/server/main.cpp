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
#include <server/websocket_server.hpp>

// WebSocket client includes
#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>
#include <thread>
#include <chrono>

/**************************************************************************************************
| WebSocket Test Client
**************************************************************************************************/

typedef websocketpp::client<websocketpp::config::asio_client> client;
typedef websocketpp::config::asio_client::message_type::ptr message_ptr;
typedef websocketpp::connection_hdl connection_hdl;

class WebSocketTestClient {
private:
    client ws_client_;
    std::thread client_thread_;
    std::atomic<bool> running_{false};
    std::string uri_;
    connection_hdl connection_;
    std::mutex connection_mutex_;
    bool connected_{false};

    void on_open(connection_hdl hdl) {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        connection_ = hdl;
        connected_ = true;
        std::cout << "🎉 [Client] Connected to WebSocket server!" << std::endl;
    }

    void on_close(connection_hdl hdl) {
        std::lock_guard<std::mutex> lock(connection_mutex_);
        connected_ = false;
        std::cout << "❌ [Client] Disconnected from WebSocket server" << std::endl;
    }

    void on_message(connection_hdl hdl, message_ptr msg) {
        std::cout << "📨 [Client] Received: " << msg->get_payload() << std::endl;

        // Send a response back every few messages (simple counter)
        static int message_count = 0;
        message_count++;

        if (message_count % 3 == 0) {
            std::string response = "Client response #" + std::to_string(message_count);
            try {
                ws_client_.send(hdl, response, websocketpp::frame::opcode::text);
                std::cout << "📤 [Client] Sent: " << response << std::endl;
            } catch (const websocketpp::exception& e) {
                std::cout << "❌ [Client] Failed to send response: " << e.what() << std::endl;
            }
        }
    }

    void on_fail(connection_hdl hdl) { std::cout << "❌ [Client] Connection failed" << std::endl; }

public:
    WebSocketTestClient(const std::string& uri) : uri_(uri) {
        // Disable all websocketpp logging for clean output
        ws_client_.clear_access_channels(websocketpp::log::alevel::all);
        ws_client_.clear_error_channels(websocketpp::log::elevel::all);

        // Initialize ASIO
        ws_client_.init_asio();

        // Set handlers
        ws_client_.set_open_handler([this](connection_hdl hdl) { on_open(hdl); });

        ws_client_.set_close_handler([this](connection_hdl hdl) { on_close(hdl); });

        ws_client_.set_message_handler([this](connection_hdl hdl, message_ptr msg) { on_message(hdl, msg); });

        ws_client_.set_fail_handler([this](connection_hdl hdl) { on_fail(hdl); });
    }

    void start() {
        running_ = true;
        client_thread_ = std::thread([this]() {
            try {
                std::cout << "🚀 [Client] Connecting to " << uri_ << "..." << std::endl;

                websocketpp::lib::error_code ec;
                client::connection_ptr con = ws_client_.get_connection(uri_, ec);

                if (ec) {
                    std::cout << "❌ [Client] Could not create connection: " << ec.message() << std::endl;
                    return;
                }

                ws_client_.connect(con);

                std::cout << "🔄 [Client] Starting event loop for 30 seconds..." << std::endl;

                // Start a timer thread to stop the client after 30 seconds
                std::thread timer_thread([this]() {
                    std::this_thread::sleep_for(std::chrono::seconds(30));
                    std::cout << "⏰ [Client] 30 second timer expired, stopping..." << std::endl;
                    running_ = false;
                    ws_client_.stop();
                });

                // Run the event loop until stopped
                ws_client_.run();

                // Wait for timer thread to finish
                if (timer_thread.joinable()) {
                    timer_thread.join();
                }

                // Close connection gracefully
                if (connected_) {
                    std::lock_guard<std::mutex> lock(connection_mutex_);
                    ws_client_.close(connection_, websocketpp::close::status::normal, "Client shutting down");
                }

            } catch (const websocketpp::exception& e) {
                std::cout << "❌ [Client] WebSocket exception: " << e.what() << std::endl;
            } catch (const std::exception& e) {
                std::cout << "❌ [Client] Exception: " << e.what() << std::endl;
            }

            std::cout << "🏁 [Client] WebSocket test client finished" << std::endl;
        });
    }

    void stop() {
        running_ = false;
        if (client_thread_.joinable()) {
            client_thread_.join();
        }
    }

    ~WebSocketTestClient() { stop(); }
};

void run_websocket_test_client(uint16_t port) {
    std::cout << "🔌 Starting WebSocket test client..." << std::endl;

    // Give the server a moment to start up
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::string uri = "ws://localhost:" + std::to_string(port);
    WebSocketTestClient client(uri);
    client.start();

    // Wait for client to finish (it runs for 30 seconds)
    std::this_thread::sleep_for(std::chrono::seconds(32));
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
        "p,port", "Port for the primary web server", cxxopts::value<int>()->default_value("8080"))(
        "ws-port", "Port for the WebSocket server", cxxopts::value<int>()->default_value("8081"))(
        "enable-websocket", "Enable WebSocket server", cxxopts::value<bool>()->default_value("false"))(
        "metal-src-dir",
        "Metal source directory (optional, defaults to TT_METAL_HOME env var)",
        cxxopts::value<std::string>())("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    bool use_mock_telemetry = result["mock-telemetry"].as<bool>();
    bool print_link_health = result["print-link-health"].as<bool>();
    int port = result["port"].as<int>();
    int ws_port = result["ws-port"].as<int>();
    bool enable_websocket = result["enable-websocket"].as<bool>();
    std::string metal_src_dir = "";
    if (result.count("metal-src-dir")) {
        metal_src_dir = result["metal-src-dir"].as<std::string>();
    }

    if (print_link_health) {
        test_print_link_health();
    }

    // Web server
    log_info(tt::LogAlways, "Starting primary web server on port {}", port);
    std::future<bool> web_server;
    std::shared_ptr<TelemetrySubscriber> web_server_subscriber;
    std::tie(web_server, web_server_subscriber) = run_web_server(port, metal_src_dir);

    // Web server #2 (testing the ability of our producer to supply two consumers)
    uint16_t secondary_port = 5555;
    log_info(tt::LogAlways, "Starting secondary web server on port {}", secondary_port);
    std::future<bool> web_server2;
    std::shared_ptr<TelemetrySubscriber> web_server2_subscriber;
    std::tie(web_server2, web_server2_subscriber) = run_web_server(secondary_port, metal_src_dir);

    // WebSocket server (optional)
    std::future<bool> websocket_server;
    std::shared_ptr<TelemetrySubscriber> websocket_subscriber;
    std::vector<std::shared_ptr<TelemetrySubscriber>> subscribers = {web_server_subscriber, web_server2_subscriber};

    // WebSocket test client thread
    std::thread websocket_client_thread;

    if (enable_websocket) {
        log_info(tt::LogAlways, "Starting WebSocket server on port {}", ws_port);
        std::tie(websocket_server, websocket_subscriber) = run_web_socket_server(ws_port, metal_src_dir);
        subscribers.push_back(websocket_subscriber);

        // Start test client in its own thread
        log_info(tt::LogAlways, "Starting WebSocket test client to connect to port {}", ws_port);
        websocket_client_thread = std::thread(run_websocket_test_client, ws_port);
    }

    if (use_mock_telemetry) {
        // Mock telemetry
        log_info(tt::LogAlways, "Using mock telemetry data");
        MockTelemetryProvider mock_telemetry(subscribers);
        mock_telemetry.run();
    } else {
        // Real telemetry
        log_info(tt::LogAlways, "Using real hardware telemetry data");
        run_telemetry_provider(subscribers);
    }

    // Run until finished
    bool web_server_succeeded = web_server.get();
    bool web_server2_succeeded = web_server2.get();
    bool websocket_succeeded = true;
    if (enable_websocket) {
        websocket_succeeded = websocket_server.get();
    }

    // Clean up WebSocket client thread
    if (enable_websocket && websocket_client_thread.joinable()) {
        log_info(tt::LogAlways, "Waiting for WebSocket test client to finish...");
        websocket_client_thread.join();
    }

    if (!web_server_succeeded || !web_server2_succeeded || !websocket_succeeded) {
        return 1;
    }

    return 0;
}
