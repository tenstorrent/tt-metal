#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * server/collection_clients.hpp
 *
 * Collection clients manager that connects to multiple endpoints and handles
 * automatic reconnection with callback-based telemetry data reception.
 */

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>

#include <telemetry/telemetry_snapshot.hpp>

/**
 * Callback function type for receiving telemetry data from WebSocket endpoints.
 *
 * @param endpoint The WebSocket endpoint URI that sent the data
 * @param snapshot The decoded telemetry snapshot received
 *
 * NOTE: This callback may be invoked from multiple threads concurrently.
 * Implementations must be thread-safe.
 */
using TelemetryCallback = std::function<void(const std::string& endpoint, const TelemetrySnapshot& snapshot)>;

// websocketpp client type
typedef websocketpp::client<websocketpp::config::asio_client> client;
typedef websocketpp::config::asio_client::message_type::ptr message_ptr;
typedef websocketpp::connection_hdl connection_hdl;

/**
 * WebSocket clients manager that connects to multiple endpoints simultaneously.
 *
 * Features:
 * - Connects to multiple WebSocket endpoints concurrently
 * - Automatic reconnection with exponential backoff on failures
 * - Single event loop for all connections (no thread-per-client)
 * - Callback-based telemetry data reception
 * - Automatic JSON parsing to TelemetrySnapshot
 * - Starts automatically in constructor, cleans up in destructor
 */
class CollectionClients {
public:
    /**
     * Create and start collection clients for the given endpoints.
     *
     * @param endpoints Vector of WebSocket URIs to connect to (e.g., "ws://localhost:8081")
     * @param callback Function to call when telemetry data is received from any endpoint
     */
    CollectionClients(const std::vector<std::string>& endpoints, TelemetryCallback callback);

    /**
     * Destructor - stops all connections and cleans up resources.
     */
    ~CollectionClients();

    // Non-copyable, non-movable
    CollectionClients(const CollectionClients&) = delete;
    CollectionClients& operator=(const CollectionClients&) = delete;
    CollectionClients(CollectionClients&&) = delete;
    CollectionClients& operator=(CollectionClients&&) = delete;

private:
    struct ConnectionInfo {
        std::string endpoint;
        connection_hdl handle;
        bool connected = false;
        std::chrono::steady_clock::time_point last_retry = std::chrono::steady_clock::now();
        int retry_count = 0;

        // Exponential backoff: 1s, 2s, 4s, 8s, max 30s
        std::chrono::seconds get_retry_delay() const {
            int delay_seconds = std::min(30, 1 << std::min(retry_count, 5));
            return std::chrono::seconds(delay_seconds);
        }
    };

    client ws_client_;
    std::vector<std::string> endpoints_;
    TelemetryCallback callback_;
    std::unordered_map<std::string, std::unique_ptr<ConnectionInfo>> connections_;
    std::mutex connections_mutex_;
    std::thread event_loop_thread_;
    std::thread retry_thread_;
    std::atomic<bool> running_{false};

    void on_open(connection_hdl hdl);
    void on_close(connection_hdl hdl);
    void on_message(connection_hdl hdl, message_ptr msg);
    void on_fail(connection_hdl hdl);
    void attempt_connection(const std::string& endpoint);
    void retry_loop();
    void event_loop();
};
