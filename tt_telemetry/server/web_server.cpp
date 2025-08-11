//TODO next: mock telemetry provider should NOT write to subscriber if it is busy. We must track busy state in release handler.
//           remove std::queue<> in consumer

#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <queue>

#include <httplib.h>

#include <server/json_messages.hpp>
#include <server/telemetry_subscriber.hpp>

using json = nlohmann::json;

class DummyTelemetrySubscriber: public TelemetrySubscriber {
    void on_telemetry_ready(HandoffHandle<TelemetrySnapshot> &&telemetry) override {
        std::cout << "Telemetry values: " << telemetry->metric_indices.size() << std::endl;
    }
};

class NewMockTelemetryProvider {
private:
    static constexpr auto UPDATE_INTERVAL_SECONDS = std::chrono::seconds(5);

    // Telemetry metrics state
    const std::vector<std::string> metric_names_ = {
        "foo_bar_baz1",
        "foo_bar_baz2",
        "foo_bar_baz3",
        "foo_glorp_cpu1",
        "foo_glorp_cpu2",
        "foo_glorp_cpu3"
    };
    std::vector<bool> metric_values_;

    // Snapshot and distribution to consumers
    std::mutex mtx_;
    std::queue<std::shared_ptr<TelemetrySnapshot>> available_buffers_;
    HandoffHandle<TelemetrySnapshot> current_buffer_;
    TelemetrySubscriber *subscriber_;
    std::atomic<bool> stopped_{false};

    // Random number generators for generating random updates
    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_int_distribution<> bool_dist_;
    std::uniform_int_distribution<> metric_dist_;
    std::uniform_int_distribution<> num_updates_dist_;

    std::thread thread_;

    void return_buffer_to_pool(std::shared_ptr<TelemetrySnapshot> buffer) {
        if (stopped_.load()) {
            return;
        }

        // Clear buffer and return to pool
        buffer->clear();
        std::lock_guard<std::mutex> lock(mtx_);
        available_buffers_.push(buffer);
    }

    HandoffHandle<TelemetrySnapshot> create_new_handoff_buffer(std::shared_ptr<TelemetrySnapshot> buffer) {
        return HandoffHandle<TelemetrySnapshot>(
            buffer,
            [this](std::shared_ptr<TelemetrySnapshot> buffer) {
                std::cout << "[MockTelemetryProvider] Released buffer" << std::endl;
                return_buffer_to_pool(buffer);
            }
        );
    }

    HandoffHandle<TelemetrySnapshot> get_writeable_buffer() {
        std::lock_guard<std::mutex> lock(mtx_);

        std::shared_ptr<TelemetrySnapshot> buffer;
        if (!available_buffers_.empty()) {
            // Get a free buffer
            buffer = available_buffers_.front();
            available_buffers_.pop();
            std::cout << "[MockTelemetryProvider] Got buffer from pool" << std::endl;
        } else {
            // Pool exhausted, create new buffer
            buffer = std::make_shared<TelemetrySnapshot>();
            std::cout << "[MockTelemetryProvider] Allocated new buffer" << std::endl;
        }

        // Return a RAII handle that will automatically return buffer to pool
        return create_new_handoff_buffer(buffer);
    }

    void create_random_updates(HandoffHandle<TelemetrySnapshot> &delta) {
        int num_updates = num_updates_dist_(gen_);
        
        // Select random endpoints to update
        for (int i = 0; i < num_updates; ++i) {
            size_t idx = metric_dist_(gen_);
            bool new_value = bool_dist_(gen_) & 1;
            
            // Only add to updates if state actually changes
            if (metric_values_[idx] != new_value) {
                metric_values_[idx] = new_value;
                delta->metric_indices.push_back(idx);
                delta->metric_values.push_back(new_value);
            }
        }

        std::cout << "[MockTelemetryProvider] Updated: " << delta->metric_indices.size() << " values pending" << std::endl;
    }

    void update_telemetry_randomly() {
        // Send initial snapshot to subscriber
        HandoffHandle<TelemetrySnapshot> snapshot = get_writeable_buffer();
        snapshot->clear();
        snapshot->is_absolute = true;
        for (size_t i = 0; i < metric_names_.size(); i++) {
            snapshot->metric_indices.push_back(i);
            snapshot->metric_names.push_back(metric_names_[i]);
            snapshot->metric_values.push_back(metric_values_[i]);
        }

        if (subscriber_ != nullptr) {
            subscriber_->on_telemetry_ready(std::move(snapshot));
        }

        // Send periodic updates
        while (!stopped_.load()) {
            if (subscriber_ != nullptr) {
                // We will now produce a delta snapshot
                HandoffHandle<TelemetrySnapshot> delta = get_writeable_buffer();
                delta->clear();
                delta->is_absolute = false;

                // Fill it with updates
                create_random_updates(delta);

                // Push to subscriber
                subscriber_->on_telemetry_ready(std::move(delta));
            }

            std::this_thread::sleep_for(UPDATE_INTERVAL_SECONDS);
        }
    }

public:
    explicit NewMockTelemetryProvider(TelemetrySubscriber *subscriber)
        : current_buffer_(create_new_handoff_buffer(std::make_shared<TelemetrySnapshot>()))
        , subscriber_(subscriber)
        , gen_(rd_())
        , bool_dist_(0, 1)
        , metric_dist_(0, metric_names_.size() - 1)
        , num_updates_dist_(1, 4) {
        // Init telemetry randomly
        size_t num_metrics = metric_names_.size();
    
        metric_values_.clear();
        metric_values_.reserve(num_metrics);
        
        for (size_t i = 0; i < num_metrics; ++i) {
            bool initial_value = bool_dist_(gen_) & 1;
            metric_values_.push_back(initial_value);
        }

        // Start update thread
        thread_ = std::thread(&NewMockTelemetryProvider::update_telemetry_randomly, this);
    }
};

class MockTelemetryProvider {
private:
    static constexpr auto UPDATE_INTERVAL = std::chrono::seconds(5);
    
    std::vector<messages::EndpointDescription> endpoints_;
    std::string host_;
    std::chrono::steady_clock::time_point last_update_time_;
    std::vector<size_t> pending_updates_indices_;
    std::vector<uint8_t> pending_updates_states_;

    // Create mock endpoints with descriptive names
    const std::vector<std::pair<std::string, std::string>> endpoint_names_ = {
        {"tray1_n1_channel0", "tray2_n1_channel0"}, // from, to
        {"tray1_n1_channel1", "tray2_n1_channel1"},
        {"tray1_n1_channel2", "tray1_n3_channel0"},
        {"tray1_n1_channel3", "tray1_n3_channel1"},
        {"tray1_n2_channel0", "tray1_n4_channel0"},
        {"tray1_n2_channel1", "tray1_n4_channel1"},
        {"tray1_n2_channel2", "tray1_n3_channel0"},
        {"tray1_n2_channel3", "tray1_n3_channel1"}
    };
    
    // Random number generation
    mutable std::random_device rd_;
    mutable std::mt19937 gen_;
    mutable std::uniform_int_distribution<> bool_dist_;
    mutable std::uniform_int_distribution<> endpoint_dist_;
    mutable std::uniform_int_distribution<> num_updates_dist_;
    
    void initialize_endpoints() {
        size_t num_endpoints = endpoint_names_.size();
    
        endpoints_.clear();
        endpoints_.reserve(num_endpoints);
        
        for (size_t i = 0; i < num_endpoints; ++i) {
            messages::EndpointDescription endpoint;
            endpoint.id = i;
            endpoint.from = endpoint_names_[i].first;
            endpoint.to = endpoint_names_[i].second;
            endpoint.state = bool_dist_(gen_) & 1;  // Random initial state
            
            endpoints_.push_back(endpoint);
        }
    }
    
    void update_random_endpoints() {
        int num_updates = num_updates_dist_(gen_);
        pending_updates_indices_.clear();
        pending_updates_states_.clear();
        
        // Select random endpoints to update
        for (int i = 0; i < num_updates; ++i) {
            size_t endpoint_idx = endpoint_dist_(gen_);
            bool new_state = bool_dist_(gen_) & 1;
            
            // Only add to updates if state actually changes
            if (endpoints_[endpoint_idx].state != new_state) {
                endpoints_[endpoint_idx].state = new_state;
                pending_updates_indices_.push_back(endpoint_idx);
                pending_updates_states_.push_back(new_state);
            }
        }
        
        last_update_time_ = std::chrono::steady_clock::now();
    }
    
public:
    explicit MockTelemetryProvider(const std::string& host = "mock-host")
        : host_(host)
        , last_update_time_(std::chrono::steady_clock::now())
        , gen_(rd_())
        , bool_dist_(0, 1)
        , endpoint_dist_(0, endpoint_names_.size() - 1)
        , num_updates_dist_(1, 4)
    {
        initialize_endpoints();
    }
    
    // Get the full snapshot of all endpoints as EndpointDefinitionMessage
    nlohmann::json get_full_snapshot() {
        messages::EndpointDefinitionMessage msg;
        msg.host = host_;
        msg.endpoints = endpoints_;
        
        nlohmann::json result;
        to_json(result, msg);
        return result;
    }
    
    // Get pending state updates as EndpointStateChangeMessage
    // Checks elapsed time and generates new updates if enough time has passed
    nlohmann::json get_updates() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = now - last_update_time_;
        
        // Generate new updates if enough time has elapsed
        if (elapsed >= UPDATE_INTERVAL) {
            update_random_endpoints();
        }
        
        // Return pending updates if any exist
        if (!pending_updates_indices_.empty()) {
            messages::EndpointStateChangeMessage msg;
            msg.host = host_;
            msg.endpoint_indices = pending_updates_indices_;
            msg.endpoint_states = pending_updates_states_;
            
            nlohmann::json result;
            to_json(result, msg);
            
            // Clear pending updates after returning them
            pending_updates_indices_.clear();
            pending_updates_states_.clear();
            
            return result;
        }
        
        return nlohmann::json::object(); // Return empty object if no updates
    }
};

class TelemetryServer: public TelemetrySubscriber {
private:
    httplib::Server server_;
    std::vector<httplib::DataSink*> sse_clients_;       // initial snapshot sent, able to receive all delta updates
    std::vector<httplib::DataSink*> new_sse_clients_;   // joined but not yet snapshotted
    std::mutex clients_mutex_;
    std::thread telemetry_thread_;
    std::atomic<bool> running_{false};
    std::chrono::time_point<std::chrono::steady_clock> started_at_;

    // Telemetry data
    std::unordered_map<size_t, std::string> metric_name_by_index_;
    std::unordered_map<size_t, bool> metric_value_by_index_;
    std::mutex snapshot_mutex_;
    std::queue<HandoffHandle<TelemetrySnapshot>> pending_snapshots_;

    void broadcast_telemetry() {
        while (running_) {
            // Get snapshot from telemetry producer thread, if one is ready
            HandoffHandle<TelemetrySnapshot> current_snapshot(nullptr, {});
            {
                std::lock_guard<std::mutex> lock(snapshot_mutex_);
                if (!pending_snapshots_.empty()) {
                    current_snapshot = std::move(pending_snapshots_.front());
                    pending_snapshots_.pop();
                }
            }

            // If no snapshot, sleep and try again
            if (!current_snapshot.is_valid()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                continue;
            }

            // Process snapshot: update internal copy of telemetry data
            //TODO: assert vectors are equal length
            if (current_snapshot->is_absolute) {
                // Absolute snapshot -- replace everything with new data
                metric_name_by_index_.clear();
                metric_value_by_index_.clear();
            }

            for (size_t i = 0; i < current_snapshot->metric_indices.size(); i++) {
                size_t idx = current_snapshot->metric_indices[i];
                if (current_snapshot->metric_names.size() > 0) {
                    // Names were included, which indicates new metrics added!
                    metric_name_by_index_.insert({ idx, current_snapshot->metric_names[i] });
                }
                metric_value_by_index_.insert({ idx, current_snapshot->metric_values[i] });
            }

            // Forward to all clients
            json j = *current_snapshot;
            std::string message = "data: " + j.dump() + "\n\n";
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

            // Handle any new clients by creating a complete snapshot for them, then add them to 
            // the permanent client list
            TelemetrySnapshot full_snapshot;
            full_snapshot.is_absolute = true;
            for (const auto &[index, name]: metric_name_by_index_) {
                full_snapshot.metric_indices.push_back(index);
                full_snapshot.metric_names.push_back(name);
                full_snapshot.metric_values.push_back(metric_value_by_index_[index]);
            }
            j = full_snapshot;
            message = "data: " + j.dump() + "\n\n";
            it = new_sse_clients_.begin();
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
    TelemetryServer()
        : started_at_(std::chrono::steady_clock::now()) 
    {
    }

    void setup_routes() {
        // Enable CORS for all routes
        server_.set_pre_routing_handler([](const httplib::Request&, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type");
            return httplib::Server::HandlerResponse::Unhandled;
        });

        // Serve static files (React app)
        server_.Get("/", [this](const httplib::Request&, httplib::Response& res) {
            std::string content = read_file("tt_telemetry/frontend/static/index.html");
            if (content.empty()) {
                res.set_content("<html><body><h1>Telemetry Server Running</h1><p>Place your React build in /static directory</p></body></html>", "text/html");
            } else {
                res.set_content(content, "text/html");
            }
        });

        // Serve static assets
        server_.Get(R"(/static/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
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
        server_.Get("/api/status", [this](const httplib::Request&, httplib::Response& res) {
            auto uptime_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at_);
            json response = {
                {"server_status", "running"},
                {"active_connections", sse_clients_.size()},
                {"uptime_seconds", uptime_seconds.count()}
            };
            res.set_content(response.dump(), "application/json");
        });

        // REST API - Get latest telemetry snapshot
        server_.Get("/api/telemetry", [this](const httplib::Request&, httplib::Response& res) {
            //TODO: return full snapshot
            //auto data = telemetry_provider_.get_full_snapshot();
            //res.set_content(data.dump(), "application/json");
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

    void start() {
        setup_routes();

        // Start telemetry broadcasting thread
        running_ = true;
        telemetry_thread_ = std::thread(&TelemetryServer::broadcast_telemetry, this);

        std::cout << "Starting telemetry server on port 8080..." << std::endl;
        std::cout << "API endpoints:" << std::endl;
        std::cout << "  GET  /                - Web UI" << std::endl;
        std::cout << "  GET  /api/status      - Server status" << std::endl;
        std::cout << "  GET  /api/telemetry   - Current telemetry" << std::endl;
        std::cout << "  GET  /api/stream      - Real-time stream (SSE)" << std::endl;

        server_.listen("0.0.0.0", 5555);
    }

    void stop() {
        running_ = false;
        server_.stop();
        if (telemetry_thread_.joinable()) {
            telemetry_thread_.join();
        }
    }

    void on_telemetry_ready(HandoffHandle<TelemetrySnapshot> &&telemetry) override {
        std::lock_guard<std::mutex> lock(snapshot_mutex_);
        pending_snapshots_.push(std::move(telemetry));
    }

    ~TelemetryServer() {
        stop();
    }
};

bool run_web_server() {
    TelemetryServer server;
    NewMockTelemetryProvider mock_provider(&server);
    
    try {
        server.start();
    } catch (const std::exception& e) {
        std::cerr << "Server error: " << e.what() << std::endl;
        return false;
    }

    return true;
}