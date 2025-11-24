// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <chrono>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <memory>
#include <set>
#include <functional>

// websocketpp includes
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include <telemetry/telemetry_subscriber.hpp>
#include <server/collection_endpoint.hpp>

using json = nlohmann::json;

// websocketpp server type
typedef websocketpp::server<websocketpp::config::asio> server;
typedef server::message_ptr message_ptr;
typedef websocketpp::connection_hdl connection_hdl;

class TelemetryCollectionEndpoint : public TelemetrySubscriber {
private:
    server ws_server_;
    std::set<connection_hdl, std::owner_less<connection_hdl>> connections_;
    std::mutex connections_mutex_;
    std::thread server_thread_;
    std::atomic<bool> running_{false};
    std::chrono::time_point<std::chrono::steady_clock> started_at_;
    uint16_t port_;

    // Accumulated telemetry data
    TelemetrySnapshot telemetry_state_;

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
        std::lock_guard<std::mutex> lock(connections_mutex_);

        for (auto it = connections_.begin(); it != connections_.end();) {
            try {
                ws_server_.send(*it, message, websocketpp::frame::opcode::text);
                ++it;
            } catch (const websocketpp::exception& e) {
                log_error(tt::LogAlways, "Failed to send message to client, removing: {}", e.what());
                it = connections_.erase(it);
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

            // Merge snapshot into telemetry state
            telemetry_state_.merge_from(*snapshot);

            // Send the delta snapshot directly to clients
            json j = *snapshot;
            std::string message = j.dump();
            send_message_to_clients(message);
        }
    }

    // WebSocket event handlers
    void on_open(connection_hdl hdl) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        connections_.insert(hdl);

        log_info(tt::LogAlways, "WebSocket client connected (total clients: {})", connections_.size());

        // Send full snapshot to the new client
        try {
            TelemetrySnapshot full_snapshot = telemetry_state_;
            json j = full_snapshot;
            std::string message = j.dump();
            ws_server_.send(hdl, message, websocketpp::frame::opcode::text);
        } catch (const websocketpp::exception& e) {
            log_error(tt::LogAlways, "Failed to send full snapshot to new client: {}", e.what());
        }
    }

    void on_close(connection_hdl hdl) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        connections_.erase(hdl);

        log_info(tt::LogAlways, "WebSocket client disconnected (total clients: {})", connections_.size());
    }

    void on_message(connection_hdl hdl, message_ptr msg) {
        // Echo the message back to the client
        try {
            ws_server_.send(hdl, msg->get_payload(), msg->get_opcode());
        } catch (const websocketpp::exception& e) {
            log_error(tt::LogAlways, "Failed to echo message: {}", e.what());
        }
    }

public:
    TelemetryCollectionEndpoint(uint16_t port) : started_at_(std::chrono::steady_clock::now()), port_(port) {
        // Disable all websocketpp logging for clean output
        ws_server_.clear_access_channels(websocketpp::log::alevel::all);
        ws_server_.clear_error_channels(websocketpp::log::elevel::all);

        // Initialize ASIO
        ws_server_.init_asio();

        // Set message handler
        ws_server_.set_message_handler([this](connection_hdl hdl, message_ptr msg) { on_message(hdl, msg); });

        // Set open handler
        ws_server_.set_open_handler([this](connection_hdl hdl) { on_open(hdl); });

        // Set close handler
        ws_server_.set_close_handler([this](connection_hdl hdl) { on_close(hdl); });
    }

    void start() {
        log_info(tt::LogAlways, "Starting WebSocket telemetry server on port {}...", port_);

        // Start telemetry processing thread
        running_ = true;
        server_thread_ = std::thread(&TelemetryCollectionEndpoint::process_telemetry, this);

        try {
            // Set reuse address
            ws_server_.set_reuse_addr(true);

            // Listen on specified port
            ws_server_.listen(port_);

            // Start the server accept loop
            ws_server_.start_accept();

            log_info(tt::LogAlways, "WebSocket server listening on port {}", port_);

            // Start the ASIO io_service run loop
            ws_server_.run();

        } catch (const websocketpp::exception& e) {
            log_error(tt::LogAlways, "WebSocket server error: {}", e.what());
            running_ = false;
        } catch (const std::exception& e) {
            log_error(tt::LogAlways, "Server error: {}", e.what());
            running_ = false;
        }

        log_info(tt::LogAlways, "WebSocket server finished");
    }

    void stop() {
        running_ = false;

        // Stop the WebSocket server
        try {
            ws_server_.stop();
        } catch (const std::exception& e) {
            log_error(tt::LogAlways, "Error stopping WebSocket server: {}", e.what());
        }

        // Wait for telemetry thread to finish
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
    }

    void on_telemetry_ready(std::shared_ptr<TelemetrySnapshot> telemetry) override {
        std::lock_guard<std::mutex> lock(snapshot_mutex_);
        pending_snapshots_.push(std::move(telemetry));
    }

    ~TelemetryCollectionEndpoint() { stop(); }
};

static bool collection_endpoint_thread(std::shared_ptr<TelemetryCollectionEndpoint> server) {
    try {
        server->start();
    } catch (const std::exception& e) {
        log_fatal(tt::LogAlways, "WebSocket server error: {}", e.what());
        return false;
    }
    return true;
}

std::pair<std::future<bool>, std::shared_ptr<TelemetrySubscriber>> run_collection_endpoint(
    uint16_t port, const std::string& metal_home) {
    auto server = std::make_shared<TelemetryCollectionEndpoint>(port);
    auto future = std::async(std::launch::async, collection_endpoint_thread, server);
    return std::make_pair(std::move(future), server);
}
