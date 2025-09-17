// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include <nlohmann/json.hpp>

#include <server/websocket_clients.hpp>

using json = nlohmann::json;

void WebSocketClients::on_open(connection_hdl hdl) {
    std::lock_guard<std::mutex> lock(connections_mutex_);

    // Find the connection info for this handle
    for (auto& [endpoint, info] : connections_) {
        if (info->handle.lock() == hdl.lock()) {
            info->connected = true;
            info->retry_count = 0;  // Reset retry count on successful connection
            std::cout << "✅ [WebSocketClients] Connected to " << endpoint << std::endl;
            break;
        }
    }
}

void WebSocketClients::on_close(connection_hdl hdl) {
    std::lock_guard<std::mutex> lock(connections_mutex_);

    // Find the connection info for this handle
    for (auto& [endpoint, info] : connections_) {
        if (info->handle.lock() == hdl.lock()) {
            info->connected = false;
            std::cout << "❌ [WebSocketClients] Disconnected from " << endpoint << std::endl;
            break;
        }
    }
}

void WebSocketClients::on_message(connection_hdl hdl, message_ptr msg) {
    std::string endpoint;

    // Find which endpoint this message came from
    {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        for (const auto& [ep, info] : connections_) {
            if (info->handle.lock() == hdl.lock()) {
                endpoint = ep;
                break;
            }
        }
    }

    if (endpoint.empty()) {
        std::cout << "⚠️  [WebSocketClients] Received message from unknown endpoint" << std::endl;
        return;
    }

    try {
        // Parse JSON message to TelemetrySnapshot
        json j = json::parse(msg->get_payload());
        TelemetrySnapshot snapshot = j.get<TelemetrySnapshot>();

        // Call the user callback
        // NOTE: This callback may be called from multiple threads concurrently
        callback_(endpoint, snapshot);

    } catch (const json::exception& e) {
        std::cout << "❌ [WebSocketClients] JSON parse error from " << endpoint << ": " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "❌ [WebSocketClients] Error processing message from " << endpoint << ": " << e.what()
                  << std::endl;
    }
}

void WebSocketClients::on_fail(connection_hdl hdl) {
    std::lock_guard<std::mutex> lock(connections_mutex_);

    // Find the connection info for this handle
    for (auto& [endpoint, info] : connections_) {
        if (info->handle.lock() == hdl.lock()) {
            info->connected = false;
            std::cout << "❌ [WebSocketClients] Connection failed to " << endpoint << std::endl;
            break;
        }
    }
}

void WebSocketClients::attempt_connection(const std::string& endpoint) {
    try {
        websocketpp::lib::error_code ec;
        client::connection_ptr con = ws_client_.get_connection(endpoint, ec);

        if (ec) {
            std::cout << "❌ [WebSocketClients] Could not create connection to " << endpoint << ": " << ec.message()
                      << std::endl;
            return;
        }

        // Store the connection handle
        {
            std::lock_guard<std::mutex> lock(connections_mutex_);
            auto& info = connections_[endpoint];
            if (!info) {
                info = std::make_unique<ConnectionInfo>();
                info->endpoint = endpoint;
            }
            info->handle = con->get_handle();
        }

        ws_client_.connect(con);
        std::cout << "🔄 [WebSocketClients] Attempting to connect to " << endpoint << std::endl;

    } catch (const std::exception& e) {
        std::cout << "❌ [WebSocketClients] Exception connecting to " << endpoint << ": " << e.what() << std::endl;
    }
}

void WebSocketClients::retry_loop() {
    while (running_) {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(connections_mutex_);

        for (auto& [endpoint, info] : connections_) {
            if (!info->connected && (now - info->last_retry) >= info->get_retry_delay()) {
                info->last_retry = now;
                info->retry_count++;

                std::cout << "🔄 [WebSocketClients] Retrying connection to " << endpoint << " (attempt "
                          << info->retry_count << ")" << std::endl;

                // Unlock mutex before attempting connection to avoid deadlock
                connections_mutex_.unlock();
                attempt_connection(endpoint);
                connections_mutex_.lock();
            }
        }
    }
}

void WebSocketClients::event_loop() {
    try {
        std::cout << "🚀 [WebSocketClients] Starting event loop..." << std::endl;
        ws_client_.run();
        std::cout << "🏁 [WebSocketClients] Event loop finished" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "❌ [WebSocketClients] Event loop exception: " << e.what() << std::endl;
    }
}

WebSocketClients::WebSocketClients(const std::vector<std::string>& endpoints, TelemetryCallback callback) :
    endpoints_(endpoints), callback_(std::move(callback)) {
    // Exit early if no endpoints provided
    if (endpoints_.empty()) {
        running_ = false;
        std::cout << "🔌 [WebSocketClients] No endpoints provided, skipping initialization" << std::endl;
        return;
    }

    // Disable all websocketpp logging for clean output
    ws_client_.clear_access_channels(websocketpp::log::alevel::all);
    ws_client_.clear_error_channels(websocketpp::log::elevel::all);

    // Initialize ASIO
    ws_client_.init_asio();

    // Set event handlers
    ws_client_.set_open_handler([this](connection_hdl hdl) { on_open(hdl); });
    ws_client_.set_close_handler([this](connection_hdl hdl) { on_close(hdl); });
    ws_client_.set_message_handler([this](connection_hdl hdl, message_ptr msg) { on_message(hdl, msg); });
    ws_client_.set_fail_handler([this](connection_hdl hdl) { on_fail(hdl); });

    // Start the system
    running_ = true;

    // Start event loop thread
    event_loop_thread_ = std::thread(&WebSocketClients::event_loop, this);

    // Start retry thread
    retry_thread_ = std::thread(&WebSocketClients::retry_loop, this);

    // Initial connection attempts
    for (const auto& endpoint : endpoints_) {
        attempt_connection(endpoint);
    }

    std::cout << "🔌 [WebSocketClients] Started with " << endpoints_.size() << " endpoints" << std::endl;
}

WebSocketClients::~WebSocketClients() {
    std::cout << "🛑 [WebSocketClients] Shutting down..." << std::endl;

    // If we exited early in constructor (no endpoints), skip cleanup
    if (endpoints_.empty()) {
        std::cout << "✅ [WebSocketClients] No cleanup needed - was not initialized" << std::endl;
        return;
    }

    running_ = false;

    // Stop the WebSocket client
    try {
        ws_client_.stop();
    } catch (const std::exception& e) {
        std::cout << "⚠️  [WebSocketClients] Error stopping client: " << e.what() << std::endl;
    }

    // Wait for threads to finish
    if (event_loop_thread_.joinable()) {
        event_loop_thread_.join();
    }
    if (retry_thread_.joinable()) {
        retry_thread_.join();
    }

    std::cout << "✅ [WebSocketClients] Shutdown complete" << std::endl;
}
