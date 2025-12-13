// PIMPL implementation for GrpcTelemetryServer
// This hides gRPC includes from the header file to prevent conflicts

#include <server/grpc_telemetry_server.hpp>

#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <grpcpp/grpcpp.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <atomic>
#include <map>

#include "telemetry_service.grpc.pb.h"
#include <utils/simple_concurrent_queue.hpp>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerWriter;
using grpc::Status;

// Forward declaration of the service impl
namespace tt {
namespace telemetry {
class TelemetryServiceImpl;
}
}  // namespace tt

// Define the PIMPL struct
struct GrpcTelemetryServer::Impl {
    std::string socket_path;
    std::unique_ptr<grpc::Server> server;
    std::unique_ptr<tt::telemetry::TelemetryServiceImpl> service_impl;

    Impl(const std::string& path) : socket_path(path) {}
};

/**************************************************************************************************
 Telemetry Service
**************************************************************************************************/

template <bool UseFilter>
static size_t populate_metrics(
    tt::telemetry::QueryMetricResponse* output,
    const TelemetrySnapshot& telemetry,
    const std::string& metric_query = "") {
    size_t count = 0;

    // Iterate through all bool metrics
    for (const auto& [path, value] : telemetry.bool_metrics) {
        // If filtering is enabled, check if metric_query is a substring of the path
        // Empty metric_query matches all metrics
        if constexpr (UseFilter) {
            if (!metric_query.empty() && path.find(metric_query) == std::string::npos) {
                continue;  // Skip this metric
            }
        }

        auto* result = output->add_bool_results();
        result->set_path(path);
        result->set_value(value);

        // Get timestamp if available
        auto ts_it = telemetry.bool_metric_timestamps.find(path);
        if (ts_it != telemetry.bool_metric_timestamps.end()) {
            result->set_timestamp(ts_it->second);
        } else {
            result->set_timestamp(0);
        }

        ++count;
    }

    // Iterate through all uint metrics
    for (const auto& [path, value] : telemetry.uint_metrics) {
        if constexpr (UseFilter) {
            if (!metric_query.empty() && path.find(metric_query) == std::string::npos) {
                continue;
            }
        }

        auto* result = output->add_uint_results();
        result->set_path(path);
        result->set_value(value);

        // Get timestamp if available
        auto ts_it = telemetry.uint_metric_timestamps.find(path);
        if (ts_it != telemetry.uint_metric_timestamps.end()) {
            result->set_timestamp(ts_it->second);
        } else {
            result->set_timestamp(0);
        }

        ++count;
    }

    // Iterate through all double metrics
    for (const auto& [path, value] : telemetry.double_metrics) {
        if constexpr (UseFilter) {
            if (!metric_query.empty() && path.find(metric_query) == std::string::npos) {
                continue;
            }
        }

        auto* result = output->add_double_results();
        result->set_path(path);
        result->set_value(value);

        // Get timestamp if available
        auto ts_it = telemetry.double_metric_timestamps.find(path);
        if (ts_it != telemetry.double_metric_timestamps.end()) {
            result->set_timestamp(ts_it->second);
        } else {
            result->set_timestamp(0);
        }

        ++count;
    }

    // Iterate through all string metrics
    for (const auto& [path, value] : telemetry.string_metrics) {
        if constexpr (UseFilter) {
            if (!metric_query.empty() && path.find(metric_query) == std::string::npos) {
                continue;
            }
        }

        auto* result = output->add_string_results();
        result->set_path(path);
        result->set_value(value);

        // Get timestamp if available
        auto ts_it = telemetry.string_metric_timestamps.find(path);
        if (ts_it != telemetry.string_metric_timestamps.end()) {
            result->set_timestamp(ts_it->second);
        } else {
            result->set_timestamp(0);
        }

        ++count;
    }

    return count;
}

namespace tt {
namespace telemetry {

// Per-client streaming state
struct StreamingClient {
    uint64_t client_id;
    std::string metric_query;
    SimpleConcurrentQueue<std::shared_ptr<TelemetrySnapshot>> update_queue;
    std::atomic<bool> active{true};
};

// Service implementation
class TelemetryServiceImpl final : public TelemetryService::Service {
public:
    TelemetryServiceImpl(const TelemetrySnapshot& telemetry_state, std::mutex& state_mutex) :
        telemetry_state_(telemetry_state), state_mutex_(state_mutex), next_client_id_(0) {}
    ~TelemetryServiceImpl() override = default;

    // Ping RPC implementation
    Status Ping(ServerContext* context, const PingRequest* request, PingResponse* response) override {
        // Echo the timestamp back to the client
        response->set_timestamp(request->timestamp());

        log_debug(tt::LogAlways, "[gRPC] Ping received with timestamp: {}", request->timestamp());

        return Status::OK;
    }

    // QueryMetric RPC implementation
    Status QueryMetric(
        ServerContext* context, const QueryMetricRequest* request, QueryMetricResponse* response) override {
        const std::string& metric_query = request->metric_query();

        log_debug(tt::LogAlways, "[gRPC] QueryMetric received for query: {}", metric_query);

        // Lock the mutex to safely access telemetry_state_
        std::lock_guard<std::mutex> lock(state_mutex_);

        // Populate response with filtered metrics and get count
        size_t num_metrics = populate_metrics<true>(response, telemetry_state_, metric_query);

        if (num_metrics == 0) {
            log_debug(tt::LogAlways, "[gRPC] No metrics found matching query: '{}'", metric_query);
            return Status(grpc::StatusCode::NOT_FOUND, "No metrics found matching query: '" + metric_query + "'");
        }

        log_debug(tt::LogAlways, "[gRPC] Found {} metric(s) matching query: '{}'", num_metrics, metric_query);
        return Status::OK;
    }

    // ListMetrics RPC implementation
    Status ListMetrics(
        ServerContext* context, const ListMetricsRequest* request, ListMetricsResponse* response) override {
        const std::string& metric_query = request->metric_query();

        log_debug(tt::LogAlways, "[gRPC] ListMetrics received with query: '{}'", metric_query);

        // Lock the mutex to safely access telemetry_state_
        std::lock_guard<std::mutex> lock(state_mutex_);

        // Add bool metric paths (with optional filtering)
        for (const auto& [path, value] : telemetry_state_.bool_metrics) {
            if (metric_query.empty() || path.find(metric_query) != std::string::npos) {
                response->add_bool_metrics(path);
            }
        }

        // Add uint metric paths (with optional filtering)
        for (const auto& [path, value] : telemetry_state_.uint_metrics) {
            if (metric_query.empty() || path.find(metric_query) != std::string::npos) {
                response->add_uint_metrics(path);
            }
        }

        // Add double metric paths (with optional filtering)
        for (const auto& [path, value] : telemetry_state_.double_metrics) {
            if (metric_query.empty() || path.find(metric_query) != std::string::npos) {
                response->add_double_metrics(path);
            }
        }

        // Add string metric paths (with optional filtering)
        for (const auto& [path, value] : telemetry_state_.string_metrics) {
            if (metric_query.empty() || path.find(metric_query) != std::string::npos) {
                response->add_string_metrics(path);
            }
        }

        [[maybe_unused]] size_t total_metrics = response->bool_metrics_size() + response->uint_metrics_size() +
                                                response->double_metrics_size() + response->string_metrics_size();
        log_debug(tt::LogAlways, "[gRPC] Returning {} total metric paths", total_metrics);

        return Status::OK;
    }

    // StreamMetrics RPC implementation (server-side streaming)
    Status StreamMetrics(
        ServerContext* context, const QueryMetricRequest* request, ServerWriter<QueryMetricResponse>* writer) override {
        const std::string& metric_query = request->metric_query();

        // Create a new streaming client with a unique ID
        uint64_t client_id = next_client_id_.fetch_add(1);
        auto client = std::make_shared<StreamingClient>();
        client->client_id = client_id;
        client->metric_query = metric_query;

        log_info(tt::LogAlways, "[gRPC] StreamMetrics: Client {} connected with query: '{}'", client_id, metric_query);

        // Register the client
        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            streaming_clients_[client_id] = client;
        }

        // Send initial snapshot of matching metrics to the client
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            QueryMetricResponse output;
            if (populate_metrics<true>(&output, telemetry_state_, metric_query) > 0) {
                if (!writer->Write(output)) {
                    log_warning(tt::LogAlways, "[gRPC] StreamMetrics: Client {} write failed!", client_id);
                }
            }
        }

        // Main streaming loop - use blocking pop_wait with timeout to efficiently wait for updates
        while (context->IsCancelled() == false) {
            // Wait up to 1 second for an update (allows checking for cancellation periodically)
            auto snapshot = client->update_queue.pop_wait(std::chrono::seconds(1));
            if (!snapshot) {
                // Timeout or queue shutdown - check if context was cancelled and continue
                continue;
            }

            QueryMetricResponse output;
            if (populate_metrics<true>(&output, **snapshot, metric_query) > 0) {
                if (!writer->Write(output)) {
                    log_warning(tt::LogAlways, "[gRPC] StreamMetrics: Client {} write failed!", client_id);
                    break;
                }
            }
        }

        // Client disconnected - clean up
        log_info(tt::LogAlways, "[gRPC] StreamMetrics: Client {} disconnected", client_id);
        client->active = false;
        client->update_queue.shutdown();  // Wake any blocked threads

        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            streaming_clients_.erase(client_id);
        }

        return Status::OK;
    }

    // Called by GrpcTelemetryServer when telemetry is updated
    // Pushes updates to all active streaming clients
    void enqueue_update_for_all_clients(const std::shared_ptr<TelemetrySnapshot>& snapshot) {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        for (const auto& [client_id, client] : streaming_clients_) {
            if (client->active) {
                client->update_queue.push(snapshot);
                log_debug(tt::LogAlways, "[gRPC] Enqueued update for streaming client {}", client_id);
            }
        }
    }

private:
    // References to the telemetry state (from GrpcTelemetryServer/TelemetrySubscriber)
    const TelemetrySnapshot& telemetry_state_;
    std::mutex& state_mutex_;

    // Streaming client management
    std::atomic<uint64_t> next_client_id_;
    std::map<uint64_t, std::shared_ptr<StreamingClient>> streaming_clients_;
    std::mutex clients_mutex_;
};

}  // namespace telemetry
}  // namespace tt

/**************************************************************************************************
 Socket Cleanup
**************************************************************************************************/

// Tracks whether the gRPC server has been started (enforces single instance, no restart)
static std::atomic<bool> g_grpc_server_started{false};

// Global socket path for signal handlers
static std::string g_grpc_socket_path;

// Signal handler for all signals (graceful shutdown and fatal crashes)
// Must be async-signal-safe - only unlink() is safe to call
static void signal_cleanup_handler(int signum) {
    // Clean up socket file (async-signal-safe, idempotent)
    unlink(g_grpc_socket_path.c_str());

    // Re-raise signal with default handler to allow normal termination/crash behavior
    std::signal(signum, SIG_DFL);
    std::raise(signum);
}

// atexit() handler for normal exit() calls (e.g., from watchdog)
static void atexit_cleanup_handler() {
    // Clean up socket file (safe in exit context, idempotent)
    unlink(g_grpc_socket_path.c_str());
}

// Install signal handlers and atexit cleanup (called once from start())
static void install_cleanup_handlers(const std::string& socket_path) {
    // Enforce single instance, no restart policy
    if (g_grpc_server_started.exchange(true)) {
        TT_FATAL(
            false,
            "GrpcTelemetryServer::start() called but server has already been started! "
            "Only one GrpcTelemetryServer instance can exist and it cannot be restarted.");
    }

    g_grpc_socket_path = socket_path;

    // Graceful shutdown signals
    std::signal(SIGINT, signal_cleanup_handler);   // ^C
    std::signal(SIGTERM, signal_cleanup_handler);  // kill command
    std::signal(SIGHUP, signal_cleanup_handler);   // terminal closed

    // Fatal crash signals
    std::signal(SIGSEGV, signal_cleanup_handler);  // Segmentation fault
    std::signal(SIGABRT, signal_cleanup_handler);  // Abort
    std::signal(SIGBUS, signal_cleanup_handler);   // Bus error
    std::signal(SIGFPE, signal_cleanup_handler);   // Floating point exception
    std::signal(SIGILL, signal_cleanup_handler);   // Illegal instruction

    // Register atexit handler for normal exit() calls (e.g., from watchdog)
    std::atexit(atexit_cleanup_handler);

    log_info(tt::LogAlways, "[gRPC] Installed signal handlers and atexit cleanup for UNIX socket");
}

/**************************************************************************************************
 Telemetry Subscriber
**************************************************************************************************/

GrpcTelemetryServer::GrpcTelemetryServer(const std::string& socket_path) :
    TelemetrySubscriber(), pimpl_(std::make_unique<Impl>(socket_path)) {
    pimpl_->service_impl = std::make_unique<tt::telemetry::TelemetryServiceImpl>(telemetry_state_, state_mutex_);
}

GrpcTelemetryServer::~GrpcTelemetryServer() { stop(); }

void GrpcTelemetryServer::start() {
    // Install signal and atexit handlers (enforces single instance, no restart)
    install_cleanup_handlers(pimpl_->socket_path);

    // Remove existing socket file if it exists
    if (unlink(pimpl_->socket_path.c_str()) == 0) {
        log_info(tt::LogAlways, "[gRPC] Removed stale UNIX socket: {}", pimpl_->socket_path);
    } else if (errno != ENOENT) {
        // ENOENT is expected (file doesn't exist), but other errors are problems
        log_warning(
            tt::LogAlways,
            "[gRPC] Failed to unlink socket {} (errno={}). This may indicate a permission issue from a previous "
            "run with elevated privileges. The socket creation may fail.",
            pimpl_->socket_path,
            errno);
    }

    ServerBuilder builder;

    // Configure to listen on UNIX socket
    // Format: unix:///path/to/socket.sock (note the triple slash for absolute path)
    std::string server_address = std::string("unix://") + pimpl_->socket_path;

    // Add listening port (no authentication for UNIX sockets)
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

    // Register the service
    builder.RegisterService(pimpl_->service_impl.get());

    // Assemble and start the server
    pimpl_->server = builder.BuildAndStart();

    if (!pimpl_->server) {
        log_error(tt::LogAlways, "[gRPC] Failed to start gRPC server on UNIX socket: {}", pimpl_->socket_path);
        return;
    }

    log_info(tt::LogAlways, "[gRPC] Server listening on UNIX socket: {}", pimpl_->socket_path);

    // Set permissions so other processes can access it
    // 0666 = read/write for owner, group, and others
    if (chmod(pimpl_->socket_path.c_str(), 0666) != 0) {
        log_warning(tt::LogAlways, "[gRPC] Failed to set permissions on socket: {}", pimpl_->socket_path);
    }
}

void GrpcTelemetryServer::stop() {
    if (pimpl_->server) {
        log_info(tt::LogAlways, "[gRPC] Shutting down gRPC telemetry server");

        // Shutdown the server with a deadline
        pimpl_->server->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(5));

        // Wait for all pending RPCs to complete
        pimpl_->server->Wait();

        pimpl_->server.reset();

        // Clean up socket file (idempotent - safe even if signal handler already called it)
        unlink(pimpl_->socket_path.c_str());

        log_info(tt::LogAlways, "[gRPC] Server stopped and socket cleaned up");
    }
}

void GrpcTelemetryServer::on_telemetry_updated(const std::shared_ptr<TelemetrySnapshot>& delta) {
    // Enqueue the update for all streaming clients
    if (pimpl_->service_impl) {
        pimpl_->service_impl->enqueue_update_for_all_clients(delta);
    }
}
