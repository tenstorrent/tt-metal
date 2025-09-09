// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <vector>

#include <httplib.h>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/assert.hpp>

#include <telemetry/telemetry_subscriber.hpp>
#include <server/web_server.hpp>

using json = nlohmann::json;

class TelemetryServer: public TelemetrySubscriber {
private:
    httplib::Server server_;
    std::vector<httplib::DataSink*> sse_clients_;       // initial snapshot sent, able to receive all delta updates
    std::vector<httplib::DataSink*> new_sse_clients_;   // joined but not yet snapshotted
    std::mutex clients_mutex_;
    std::thread telemetry_thread_;
    std::atomic<bool> running_{false};
    std::chrono::time_point<std::chrono::steady_clock> started_at_;
    std::string metal_home_;

    // Telemetry data
    std::unordered_map<size_t, std::string> bool_metric_name_by_id_;
    std::unordered_map<size_t, bool> bool_metric_value_by_id_;
    std::unordered_map<size_t, std::string> uint_metric_name_by_id_;
    std::unordered_map<size_t, uint16_t> uint_metric_units_by_id_;
    std::unordered_map<size_t, uint64_t> uint_metric_value_by_id_;
    std::unordered_map<size_t, std::string> double_metric_name_by_id_;
    std::unordered_map<size_t, uint16_t> double_metric_units_by_id_;
    std::unordered_map<size_t, double> double_metric_value_by_id_;
    std::unordered_map<uint16_t, std::string> metric_unit_display_label_by_code_;
    std::unordered_map<uint16_t, std::string> metric_unit_full_label_by_code_;
    std::mutex snapshot_mutex_;
    std::queue<std::shared_ptr<TelemetrySnapshot>> pending_snapshots_;

    // Send snapshot to all new clients and move those clients to the client list. This must be run
    // from the main client thread.
    void handle_new_clients() {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        if (new_sse_clients_.size() == 0) {
            return;
        }

        // Construct snapshot from current data
        TelemetrySnapshot full_snapshot;
        for (const auto &[id, name]: bool_metric_name_by_id_) {
            full_snapshot.bool_metric_ids.push_back(id);
            full_snapshot.bool_metric_names.push_back(name);
            full_snapshot.bool_metric_values.push_back(bool_metric_value_by_id_[id]);
        }
        for (const auto &[id, name]: uint_metric_name_by_id_) {
            full_snapshot.uint_metric_ids.push_back(id);
            full_snapshot.uint_metric_names.push_back(name);
            full_snapshot.uint_metric_units.push_back(uint_metric_units_by_id_[id]);
            full_snapshot.uint_metric_values.push_back(uint_metric_value_by_id_[id]);
        }
        for (const auto& [id, name] : double_metric_name_by_id_) {
            full_snapshot.double_metric_ids.push_back(id);
            full_snapshot.double_metric_names.push_back(name);
            full_snapshot.double_metric_units.push_back(double_metric_units_by_id_[id]);
            full_snapshot.double_metric_values.push_back(double_metric_value_by_id_[id]);
        }

        // Include cached unit label maps
        full_snapshot.metric_unit_display_label_by_code = metric_unit_display_label_by_code_;
        full_snapshot.metric_unit_full_label_by_code = metric_unit_full_label_by_code_;
        json j = full_snapshot;
        std::string message = "data: " + j.dump() + "\n\n";

        // Send to all new clients and move those clients to the client list
        auto it = new_sse_clients_.begin();
        while (it != new_sse_clients_.end()) {
            if (!(*it)->write(message.c_str(), message.size())) {
                // Client disconnected
                it = new_sse_clients_.erase(it);
            } else {
                // Add client
                sse_clients_.push_back(*it);
                ++it;
            }
        }
        new_sse_clients_.clear();
    }

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

    void update_telemetry_state_from_snapshot(std::shared_ptr<TelemetrySnapshot> snapshot) {
        TT_ASSERT(snapshot->bool_metric_ids.size() == snapshot->bool_metric_values.size());
        if (snapshot->bool_metric_names.size() > 0) {
            TT_ASSERT(snapshot->bool_metric_ids.size() == snapshot->bool_metric_names.size());
        }
        TT_ASSERT(snapshot->uint_metric_ids.size() == snapshot->uint_metric_values.size());
        if (snapshot->uint_metric_names.size() > 0) {
            TT_ASSERT(snapshot->uint_metric_ids.size() == snapshot->uint_metric_names.size());
            TT_ASSERT(snapshot->uint_metric_ids.size() == snapshot->uint_metric_units.size());
        }
        TT_ASSERT(snapshot->double_metric_ids.size() == snapshot->double_metric_values.size());
        if (snapshot->double_metric_names.size() > 0) {
            TT_ASSERT(snapshot->double_metric_ids.size() == snapshot->double_metric_names.size());
            TT_ASSERT(snapshot->double_metric_ids.size() == snapshot->double_metric_units.size());
        }

        // Cache unit label maps when any names are populated
        if (snapshot->uint_metric_names.size() > 0 || snapshot->double_metric_names.size() > 0) {
            metric_unit_display_label_by_code_ = snapshot->metric_unit_display_label_by_code;
            metric_unit_full_label_by_code_ = snapshot->metric_unit_full_label_by_code;
        }

        for (size_t i = 0; i < snapshot->bool_metric_ids.size(); i++) {
            size_t idx = snapshot->bool_metric_ids[i];
            if (snapshot->bool_metric_names.size() > 0) {
                // Names were included, which indicates new metrics added!
                bool_metric_name_by_id_[idx] = snapshot->bool_metric_names[i];
            }
            bool_metric_value_by_id_[idx] = snapshot->bool_metric_values[i];
        }

        for (size_t i = 0; i < snapshot->uint_metric_ids.size(); i++) {
            size_t idx = snapshot->uint_metric_ids[i];
            if (snapshot->uint_metric_names.size() > 0) {
                // Names were included, which indicates new metrics added!
                uint_metric_name_by_id_[idx] = snapshot->uint_metric_names[i];
                uint_metric_units_by_id_[idx] = snapshot->uint_metric_units[i];
            }
            uint_metric_value_by_id_[idx] = snapshot->uint_metric_values[i];
        }

        for (size_t i = 0; i < snapshot->double_metric_ids.size(); i++) {
            size_t idx = snapshot->double_metric_ids[i];
            if (snapshot->double_metric_names.size() > 0) {
                // Names were included, which indicates new metrics added!
                double_metric_name_by_id_[idx] = snapshot->double_metric_names[i];
                double_metric_units_by_id_[idx] = snapshot->double_metric_units[i];
            }
            double_metric_value_by_id_[idx] = snapshot->double_metric_values[i];
        }
    }

    void send_snapshot_to_clients(std::shared_ptr<TelemetrySnapshot> snapshot) {
        // Serialize
        json j = *snapshot;
        std::string message = "data: " + j.dump() + "\n\n";

        // Send to all, removing any disconnected clients
        std::lock_guard<std::mutex> lock(clients_mutex_);
        auto it = sse_clients_.begin();
        while (it != sse_clients_.end()) {
            if (!(*it)->write(message.c_str(), message.size())) {
                // Client disconnected
                it = sse_clients_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Main client thread
    void broadcast_telemetry() {
        while (running_) {
            handle_new_clients();
            std::shared_ptr<TelemetrySnapshot> snapshot = get_next_snapshot();
            if (!snapshot) {
                // No snapshot, sleep a while
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                continue;
            }
            update_telemetry_state_from_snapshot(snapshot);
            send_snapshot_to_clients(snapshot);
        }
    }

    std::string join_paths(const std::string& base, const std::string& path) {
        if (base.empty()) {
            return path;
        }
        if (path.empty()) {
            return base;
        }

        // Check if base ends with separator
        bool base_has_separator = (base.back() == '/' || base.back() == '\\');
        // Check if path starts with separator
        bool path_has_separator = (path.front() == '/' || path.front() == '\\');

        if (base_has_separator && path_has_separator) {
            // Both have separator, remove one
            return base + path.substr(1);
        } else if (!base_has_separator && !path_has_separator) {
            // Neither has separator, add one
            return base + "/" + path;
        } else {
            // One has separator, just concatenate
            return base + path;
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
    TelemetryServer(const std::string& metal_home = "") : started_at_(std::chrono::steady_clock::now()) {
        if (!metal_home.empty()) {
            metal_home_ = metal_home;
        } else {
            const char* env_metal_home = std::getenv("TT_METAL_HOME");
            if (env_metal_home != nullptr) {
                metal_home_ = std::string(env_metal_home);
            } else {
                throw std::runtime_error("TT_METAL_HOME environment variable not set and no metal_home provided");
            }
        }
    }

    void setup_routes() {
        // Enable CORS for all routes
        server_.set_pre_routing_handler([](const httplib::Request&, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type");
            return httplib::Server::HandlerResponse::Unhandled;
        });

        // Serve static assets from root path (/<foo> maps to $(TT_METAL_HOME)/tt_telemetry/frontend/static/<foo>)
        // This includes serving index.html for the root path "/"
        server_.Get(R"(^/(?!api/)(.*))", [this](const httplib::Request& req, httplib::Response& res) {
            std::string path = req.matches[1].str();

            // If path is empty (root "/"), serve index.html
            if (path.empty()) {
                path = "index.html";
            }

            std::string content = read_file(join_paths(metal_home_, "tt_telemetry/frontend/static/" + path));
            if (!content.empty()) {
                // Set appropriate content type
                if (path.ends_with(".html") || path.ends_with(".htm")) {
                    res.set_content(content, "text/html");
                } else if (path.ends_with(".js")) {
                    res.set_content(content, "application/javascript");
                } else if (path.ends_with(".css")) {
                    res.set_content(content, "text/css");
                } else if (path.ends_with(".json")) {
                    res.set_content(content, "application/json");
                } else if (
                    path.ends_with(".png") || path.ends_with(".jpg") || path.ends_with(".jpeg") ||
                    path.ends_with(".gif")) {
                    res.set_content(content, "image/*");
                } else if (path.ends_with(".svg")) {
                    res.set_content(content, "image/svg+xml");
                } else if (path.ends_with(".ico")) {
                    res.set_content(content, "image/x-icon");
                } else {
                    res.set_content(content, "application/octet-stream");
                }
            } else {
                // If file not found, serve index.html for SPA routing
                std::string index_content =
                    read_file(join_paths(metal_home_, "tt_telemetry/frontend/static/index.html"));
                if (!index_content.empty()) {
                    res.set_content(index_content, "text/html");
                } else {
                    res.status = 404;
                    res.set_content(
                        "<html><body><h1>Telemetry Server Running</h1><p>404: File not found</p></body></html>",
                        "text/html");
                }
            }
        });

        // REST API - Get current system status
        server_.Get("/api/status", [this](const httplib::Request&, httplib::Response& res) {
            auto uptime_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at_);
            json response = {
                {"server_status", "running"},
                {"active_connections", sse_clients_.size()},
                {"uptime_seconds", uptime_seconds.count()}
            };
            res.set_content(response.dump(), "application/json");
        });

        // Server-Sent Events endpoint for real-time telemetry
        server_.Get("/api/stream", [this](const httplib::Request&, httplib::Response& res) {
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");

            res.set_content_provider(
                "text/event-stream",
                [this](size_t /*offset*/, httplib::DataSink& sink) {
                    // Add to new client list for initial snapshot and future updates
                    {
                        std::lock_guard<std::mutex> lock(clients_mutex_);
                        new_sse_clients_.push_back(&sink);
                    }

                    // Keep connection alive - the broadcast_telemetry thread will send updates
                    // We'll rely on the write operations to detect disconnection
                    while (running_) {
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                        // The broadcast_telemetry thread handles actual data sending
                        // If client disconnects, write() will fail and remove the client
                    }

                    // Remove client when connection ends
                    {
                        std::lock_guard<std::mutex> lock(clients_mutex_);
                        sse_clients_.erase(
                            std::remove(sse_clients_.begin(), sse_clients_.end(), &sink),
                            sse_clients_.end());
                        new_sse_clients_.erase(
                            std::remove(new_sse_clients_.begin(), new_sse_clients_.end(), &sink),
                            new_sse_clients_.end());
                    }

                    std::cout << "Connection finished" << std::endl;

                    return true;
                }
            );
        });

        // Handle OPTIONS requests for CORS
        server_.Options(".*", [](const httplib::Request&, httplib::Response&) {
            return;
        });
    }

    void start(uint16_t port) {
        setup_routes();

        // Start telemetry broadcasting thread
        running_ = true;
        telemetry_thread_ = std::thread(&TelemetryServer::broadcast_telemetry, this);

        std::cout << "Starting telemetry server on port " << port << "..." << std::endl;
        std::cout << "API endpoints:" << std::endl;
        std::cout << "  GET  /                - Web UI (serves static/index.html)" << std::endl;
        std::cout << "  GET  /<path>          - Static assets (serves static/<path>)" << std::endl;
        std::cout << "  GET  /api/status      - Server status" << std::endl;
        std::cout << "  GET  /api/telemetry   - Current telemetry" << std::endl;
        std::cout << "  GET  /api/stream      - Real-time stream (SSE)" << std::endl;

        server_.listen("0.0.0.0", port);
    }

    void stop() {
        running_ = false;
        server_.stop();
        if (telemetry_thread_.joinable()) {
            telemetry_thread_.join();
        }
    }

    void on_telemetry_ready(std::shared_ptr<TelemetrySnapshot> telemetry) override {
        std::lock_guard<std::mutex> lock(snapshot_mutex_);
        pending_snapshots_.push(std::move(telemetry));
    }

    ~TelemetryServer() {
        stop();
    }
};

static bool web_server_thread(std::shared_ptr<TelemetryServer> server, uint16_t port) {
    try {
        server->start(port);
    } catch (const std::exception& e) {
        log_fatal(tt::LogAlways, "Web server error: {}", e.what());
        return false;
    }
    return true;
}

std::pair<std::future<bool>, std::shared_ptr<TelemetrySubscriber>> run_web_server(
    uint16_t port, const std::string& metal_home) {
    auto server = std::make_shared<TelemetryServer>(metal_home);
    auto subscriber = static_pointer_cast<TelemetrySubscriber>(server);
    auto future = std::async(std::launch::async, web_server_thread, server, port);
    return std::make_pair(std::move(future), subscriber);
}
