// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <memory>
#include <set>

#include <App.h>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/assert.hpp>

#include <telemetry/telemetry_subscriber.hpp>
#include <server/websocket_server.hpp>

using json = nlohmann::json;

class TelemetryWebSocketServer : public TelemetrySubscriber {
private:
    struct PerSocketData {
        // TODO
    };

    std::set<uWS::WebSocket<false, true, PerSocketData>*> clients_;
    std::mutex clients_mutex_;
    std::thread server_thread_;
    std::atomic<bool> running_{false};
    std::chrono::time_point<std::chrono::steady_clock> started_at_;
    uint16_t port_;

    // Telemetry data
    std::mutex snapshot_mutex_;
    std::queue<std::shared_ptr<TelemetrySnapshot>> pending_snapshots_;

    // Get snapshot if one is ready
    std::shared_ptr<TelemetrySnapshot> get_next_snapshot() {
        std::lock_guard<std::mutex> lock(snapshot_mutex_);

        std::shared_ptr<TelemetrySnapshot> snapshot;
        if (!pending_snapshots_.empty()) {
            snapshot = std::move(pending_snapshots_.front());
            pending_snapshots_.pop();
        }
        return snapshot;
    }

    void send_message_to_clients(const std::string& message) {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        auto it = clients_.begin();
        while (it != clients_.end()) {
            auto* ws = *it;
            auto result = ws->send(message, uWS::OpCode::TEXT);
            if (result == uWS::WebSocket<false, true, PerSocketData>::SendStatus::DROPPED) {
                // Client disconnected, remove from set
                it = clients_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Main telemetry processing thread
    void process_telemetry() {
        while (running_) {
            std::shared_ptr<TelemetrySnapshot> snapshot = get_next_snapshot();
            if (!snapshot) {
                // No snapshot, sleep a while
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                continue;
            }

            // For now, just send a simple message indicating we received telemetry
            // In the future, this would serialize and send the actual telemetry data
            json j = *snapshot;
            std::string message = j.dump();
            send_message_to_clients(message);
        }
    }

public:
    TelemetryWebSocketServer(uint16_t port) : started_at_(std::chrono::steady_clock::now()), port_(port) {}

    void start() {
        std::cout << "Starting WebSocket telemetry server on port " << port_ << "..." << std::endl;

        // Start telemetry processing thread
        running_ = true;
        server_thread_ = std::thread(&TelemetryWebSocketServer::process_telemetry, this);

        // Run WebSocket server until completion
        uWS::App()
            .ws<PerSocketData>(
                "/*",
                {/* Settings */
                 .compression = uWS::CompressOptions(uWS::DEDICATED_COMPRESSOR | uWS::DEDICATED_DECOMPRESSOR),
                 .maxPayloadLength = 100 * 1024 * 1024,
                 .idleTimeout = 16,
                 .maxBackpressure = 100 * 1024 * 1024,
                 .closeOnBackpressureLimit = false,
                 .resetIdleTimeoutOnSend = false,
                 .sendPingsAutomatically = true,
                 /* Handlers */
                 .upgrade = nullptr,
                 .open =
                     [this](auto* ws) {
                         std::cout << "WebSocket client connected from " << ws->getRemoteAddressAsText() << std::endl;

                         // Add client to our set
                         {
                             std::lock_guard<std::mutex> lock(clients_mutex_);
                             clients_.insert(ws);
                         }

                         // Send hello message to the new client
                         ws->send("hello", uWS::OpCode::TEXT);
                     },
                 .message =
                     [](auto* ws, std::string_view message, uWS::OpCode opCode) {
                         /* This is the opposite of what you probably want; compress if message is LARGER than 16 kb
                          * the reason we do the opposite here; compress if SMALLER than 16 kb is to allow for
                          * benchmarking of large message sending without compression */

                         /* Never mind, it changed back to never compressing for now */
                         ws->send(message, opCode, false);
                     },
                 .dropped =
                     [](auto* /*ws*/, std::string_view /*message*/, uWS::OpCode /*opCode*/) {
                         /* A message was dropped due to set maxBackpressure and closeOnBackpressureLimit limit */
                     },
                 .drain =
                     [](auto* /*ws*/) {
                         /* Check ws->getBufferedAmount() here */
                     },
                 .ping =
                     [](auto* /*ws*/, std::string_view) {
                         /* Not implemented yet */
                     },
                 .pong =
                     [](auto* /*ws*/, std::string_view) {
                         /* Not implemented yet */
                     },
                 .close =
                     [this](auto* ws, int /*code*/, std::string_view /*message*/) {
                         std::cout << "WebSocket client disconnected" << std::endl;

                         // Remove client from our set
                         std::lock_guard<std::mutex> lock(clients_mutex_);
                         clients_.erase(ws);
                     }})
            .get(
                "/test",
                [](auto* res, auto* req) {
                    std::cout << "🌐 HTTP test route accessed" << std::endl;
                    res->end("WebSocket server is running");
                })
            .listen(
                port_,
                [this](auto* listen_socket) {
                    if (listen_socket) {
                        std::cout << "Listening on port " << port_ << std::endl;
                    } else {
                        std::cout << "Failed to bind to port " << port_ << std::endl;
                        running_ = false;
                    }
                })
            .run();

        std::cout << "WebSocket server finished" << std::endl;
    }

    void stop() {
        // TODO: we currently have no way of stopping the WebSocket server!
        running_ = false;
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
    }

    void on_telemetry_ready(std::shared_ptr<TelemetrySnapshot> telemetry) override {
        std::lock_guard<std::mutex> lock(snapshot_mutex_);
        pending_snapshots_.push(std::move(telemetry));
    }

    ~TelemetryWebSocketServer() { stop(); }
};

static bool websocket_server_thread(std::shared_ptr<TelemetryWebSocketServer> server) {
    try {
        server->start();
    } catch (const std::exception& e) {
        log_fatal(tt::LogAlways, "WebSocket server error: {}", e.what());
        return false;
    }
    return true;
}

std::pair<std::future<bool>, std::shared_ptr<TelemetrySubscriber>> run_web_socket_server(
    uint16_t port, const std::string& metal_home) {
    auto server = std::make_shared<TelemetryWebSocketServer>(port);
    auto future = std::async(std::launch::async, websocket_server_thread, server);
    return std::make_pair(std::move(future), server);
}
