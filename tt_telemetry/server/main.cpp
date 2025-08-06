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

#include <httplib.h>

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


/**************************************************************************************************
 Web Server
**************************************************************************************************/

using json = nlohmann::json;
class TelemetryServer {
private:
    httplib::Server server;
    std::vector<httplib::DataSink*> sse_clients;
    std::mutex clients_mutex;
    std::thread telemetry_thread;
    std::atomic<bool> running{false};

    // Mock telemetry data generator
    json generate_telemetry_data() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> cpu_dist(0.0, 100.0);
        static std::uniform_real_distribution<> mem_dist(40.0, 90.0);
        static std::uniform_real_distribution<> temp_dist(30.0, 80.0);

        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();

        return json{
            {"timestamp", timestamp},
            {"cpu_usage", cpu_dist(gen)},
            {"memory_usage", mem_dist(gen)},
            {"temperature", temp_dist(gen)},
            {"network_rx", static_cast<int>(cpu_dist(gen) * 1000)},
            {"network_tx", static_cast<int>(cpu_dist(gen) * 800)}
        };
    }

    void broadcast_telemetry() {
        while (running) {
            auto data = generate_telemetry_data();
            std::string message = "data: " + data.dump() + "\n\n";

            std::lock_guard<std::mutex> lock(clients_mutex);
            auto it = sse_clients.begin();
            while (it != sse_clients.end()) {
                if (!(*it)->write(message.c_str(), message.size())) {
                    // Client disconnected
                    it = sse_clients.erase(it);
                } else {
                    ++it;
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }

    std::string read_file(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            return "";
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

public:
    void setup_routes() {
        // Enable CORS for all routes
        server.set_pre_routing_handler([](const httplib::Request&, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type");
            return httplib::Server::HandlerResponse::Unhandled;
        });

        // Serve static files (React app)
        server.Get("/", [this](const httplib::Request&, httplib::Response& res) {
            std::string content = read_file("tt_telemetry/frontend/static/index.html");
            if (content.empty()) {
                res.set_content("<html><body><h1>Telemetry Server Running</h1><p>Place your React build in /static directory</p></body></html>", "text/html");
            } else {
                res.set_content(content, "text/html");
            }
        });

        // Serve static assets
        server.Get(R"(/static/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
            std::string filename = req.matches[1];
            std::string content = read_file("tt_telemetry/frontend/static/" + filename);
            if (!content.empty()) {
                // Set appropriate content type
                if (filename.ends_with(".html") || filename.ends_with(".htm")) {
                    res.set_content(content, "text/html");
                } else if (filename.ends_with(".js")) {
                    res.set_content(content, "application/javascript");
                } else if (filename.ends_with(".css")) {
                    res.set_content(content, "text/css");
                } else if (filename.ends_with(".json")) {
                    res.set_content(content, "application/json");
                } else {
                    res.set_content(content, "application/octet-stream");
                }
            } else {
                res.status = 404;
            }
        });

        // REST API - Get current system status
        server.Get("/api/status", [this](const httplib::Request&, httplib::Response& res) {
            json response = {
                {"server_status", "running"},
                {"active_connections", sse_clients.size()},
                {"uptime_seconds", 12345}, // Mock uptime
                {"version", "1.0.0"}
            };
            res.set_content(response.dump(), "application/json");
        });

        // REST API - Get latest telemetry snapshot
        server.Get("/api/telemetry", [this](const httplib::Request&, httplib::Response& res) {
            auto data = generate_telemetry_data();
            res.set_content(data.dump(), "application/json");
        });

        // Server-Sent Events endpoint for real-time telemetry
        server.Get("/api/stream", [this](const httplib::Request&, httplib::Response& res) {
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");

            res.set_content_provider(
                "text/event-stream",
                [this](size_t /*offset*/, httplib::DataSink& sink) {
                    {
                        std::lock_guard<std::mutex> lock(clients_mutex);
                        sse_clients.push_back(&sink);
                    }

                    // Send initial data
                    auto initial_data = generate_telemetry_data();
                    std::string initial_message = "data: " + initial_data.dump() + "\n\n";
                    sink.write(initial_message.c_str(), initial_message.size());

                    // Keep connection alive - the broadcast_telemetry thread will send updates
                    // We'll rely on the write operations to detect disconnection
                    while (running) {
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                        // The broadcast_telemetry thread handles actual data sending
                        // If client disconnects, write() will fail and remove the client
                    }

                    // Remove client when connection ends
                    {
                        std::lock_guard<std::mutex> lock(clients_mutex);
                        sse_clients.erase(
                            std::remove(sse_clients.begin(), sse_clients.end(), &sink),
                            sse_clients.end());
                    }

                    return true;
                }
            );
        });

        // Handle OPTIONS requests for CORS
        server.Options(".*", [](const httplib::Request&, httplib::Response&) {
            return;
        });
    }

    void start() {
        setup_routes();

        // Start telemetry broadcasting thread
        running = true;
        telemetry_thread = std::thread(&TelemetryServer::broadcast_telemetry, this);

        std::cout << "Starting telemetry server on port 8080..." << std::endl;
        std::cout << "API endpoints:" << std::endl;
        std::cout << "  GET  /                - Web UI" << std::endl;
        std::cout << "  GET  /api/status      - Server status" << std::endl;
        std::cout << "  GET  /api/telemetry   - Current telemetry" << std::endl;
        std::cout << "  GET  /api/stream      - Real-time stream (SSE)" << std::endl;

        server.listen("0.0.0.0", 5555);
    }

    void stop() {
        running = false;
        server.stop();
        if (telemetry_thread.joinable()) {
            telemetry_thread.join();
        }
    }

    ~TelemetryServer() {
        stop();
    }
};

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

    // Web server
    TelemetryServer server;
    
    try {
        server.start();
    } catch (const std::exception& e) {
        std::cerr << "Server error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}