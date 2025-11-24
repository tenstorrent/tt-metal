// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TODO: check for timestamp should be unnecessary and can be assumed to be present if metric is.
//       eventually, we should refactor TelemetrySnapshot() to provide a getter that validates all this.

#include <server/grpc_telemetry_server.hpp>

#include <sys/stat.h>
#include <unistd.h>
#include <grpcpp/grpcpp.h>
#include <tt-logger/tt-logger.hpp>

#include "telemetry_service.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

// Helper function to extract metric name from a path
// Returns the substring after the last '/', or the entire string if no '/' is found
static std::string_view get_metric_name_from_path(std::string_view path) {
    auto last_slash = path.find_last_of('/');
    if (last_slash == std::string_view::npos) {
        return path;
    }
    return path.substr(last_slash + 1);
}

static void update_name_to_paths_mapping(
    std::unordered_map<std::string, std::unordered_set<std::string>>* paths_by_name,
    const TelemetrySnapshot& snapshot) {
    // It's gross that we have to do this each time for *all* metrics in the snapshot. Eventually,
    // we will break up TelemetrySnapshots into updates and new additions, so we only need to run
    // this on newly-added metrics.
    for (const auto& key : std::views::keys(snapshot.bool_metrics)) {
        (*paths_by_name)[std::string(get_metric_name_from_path(key))].insert(key);
    }
    for (const auto& key : std::views::keys(snapshot.uint_metrics)) {
        (*paths_by_name)[std::string(get_metric_name_from_path(key))].insert(key);
    }
    for (const auto& key : std::views::keys(snapshot.double_metrics)) {
        (*paths_by_name)[std::string(get_metric_name_from_path(key))].insert(key);
    }
    for (const auto& key : std::views::keys(snapshot.string_metrics)) {
        (*paths_by_name)[std::string(get_metric_name_from_path(key))].insert(key);
    }
}

namespace tt {
namespace telemetry {

// Service implementation
class TelemetryServiceImpl final : public TelemetryService::Service {
public:
    TelemetryServiceImpl(
        const TelemetrySnapshot& telemetry_state,
        const std::unordered_map<std::string, std::unordered_set<std::string>>& metric_paths_by_name,
        std::mutex& state_mutex) :
        telemetry_state_(telemetry_state), metric_paths_by_name_(metric_paths_by_name), state_mutex_(state_mutex) {}
    ~TelemetryServiceImpl() override = default;

    // Ping RPC implementation
    Status Ping(ServerContext* context, const PingRequest* request, PingResponse* response) override {
        // Echo the timestamp back to the client
        response->set_timestamp(request->timestamp());

        log_debug(tt::LogAlways, "gRPC Ping received with timestamp: {}", request->timestamp());

        return Status::OK;
    }

    // QueryMetric RPC implementation
    Status QueryMetric(
        ServerContext* context, const QueryMetricRequest* request, QueryMetricResponse* response) override {
        const std::string& metric_query = request->metric_query();

        log_debug(tt::LogAlways, "gRPC QueryMetric received for metric: {}", metric_query);

        bool found_any = false;

        // Lock the mutex to safely access telemetry_state_ and metric_name_to_paths_
        std::lock_guard<std::mutex> lock(state_mutex_);

        // Get all metric paths that contain the metric name
        auto it = metric_paths_by_name_.find(metric_query);
        if (it != metric_paths_by_name_.end()) {
            for (const auto& metric_path : it->second) {
                // Check each metric type map to find the requested metric
                // Try bool metrics first
                auto bool_it = telemetry_state_.bool_metrics.find(metric_path);
                if (bool_it != telemetry_state_.bool_metrics.end()) {
                    auto* result = response->add_bool_results();
                    result->set_path(metric_path);
                    result->set_value(bool_it->second);

                    // Get timestamp if available
                    auto ts_it = telemetry_state_.bool_metric_timestamps.find(metric_path);
                    if (ts_it != telemetry_state_.bool_metric_timestamps.end()) {
                        result->set_timestamp(ts_it->second);
                    } else {
                        result->set_timestamp(0);
                    }

                    log_debug(tt::LogAlways, "Found bool metric '{}' with value: {}", metric_path, bool_it->second);
                    found_any = true;
                }

                // Try uint metrics
                auto uint_it = telemetry_state_.uint_metrics.find(metric_path);
                if (uint_it != telemetry_state_.uint_metrics.end()) {
                    auto* result = response->add_uint_results();
                    result->set_path(metric_path);
                    result->set_value(uint_it->second);

                    // Get timestamp if available
                    auto ts_it = telemetry_state_.uint_metric_timestamps.find(metric_path);
                    if (ts_it != telemetry_state_.uint_metric_timestamps.end()) {
                        result->set_timestamp(ts_it->second);
                    } else {
                        result->set_timestamp(0);
                    }

                    log_debug(tt::LogAlways, "Found uint metric '{}' with value: {}", metric_path, uint_it->second);
                    found_any = true;
                }

                // Try double metrics
                auto double_it = telemetry_state_.double_metrics.find(metric_path);
                if (double_it != telemetry_state_.double_metrics.end()) {
                    auto* result = response->add_double_results();
                    result->set_path(metric_path);
                    result->set_value(double_it->second);

                    // Get timestamp if available
                    auto ts_it = telemetry_state_.double_metric_timestamps.find(metric_path);
                    if (ts_it != telemetry_state_.double_metric_timestamps.end()) {
                        result->set_timestamp(ts_it->second);
                    } else {
                        result->set_timestamp(0);
                    }

                    log_debug(tt::LogAlways, "Found double metric '{}' with value: {}", metric_path, double_it->second);
                    found_any = true;
                }

                // Try string metrics
                auto string_it = telemetry_state_.string_metrics.find(metric_path);
                if (string_it != telemetry_state_.string_metrics.end()) {
                    auto* result = response->add_string_results();
                    result->set_path(metric_path);
                    result->set_value(string_it->second);

                    // Get timestamp if available
                    auto ts_it = telemetry_state_.string_metric_timestamps.find(metric_path);
                    if (ts_it != telemetry_state_.string_metric_timestamps.end()) {
                        result->set_timestamp(ts_it->second);
                    } else {
                        result->set_timestamp(0);
                    }

                    log_debug(tt::LogAlways, "Found string metric '{}' with value: {}", metric_path, string_it->second);
                    found_any = true;
                }
            }
        }

        // Metric not found in any map - return error status
        if (!found_any) {
            log_debug(tt::LogAlways, "Metric '{}' not found in telemetry state", metric_query);
            return Status(grpc::StatusCode::NOT_FOUND, "Metric '" + metric_query + "' not found");
        }

        return Status::OK;
    }

private:
    // References to the telemetry state (from GrpcTelemetryServer/TelemetrySubscriber)
    const TelemetrySnapshot& telemetry_state_;
    const std::unordered_map<std::string, std::unordered_set<std::string>>& metric_paths_by_name_;
    std::mutex& state_mutex_;
};

}  // namespace telemetry
}  // namespace tt

GrpcTelemetryServer::GrpcTelemetryServer() :
    TelemetrySubscriber(),
    service_impl_(
        std::make_unique<tt::telemetry::TelemetryServiceImpl>(telemetry_state_, metric_paths_by_name_, state_mutex_)) {}

GrpcTelemetryServer::~GrpcTelemetryServer() { stop(); }

void GrpcTelemetryServer::start() {
    // Remove existing socket file if it exists
    unlink(GRPC_TELEMETRY_SOCKET_PATH);

    ServerBuilder builder;

    // Configure to listen on UNIX socket
    // Format: unix:///path/to/socket.sock (note the triple slash for absolute path)
    std::string server_address = std::string("unix://") + GRPC_TELEMETRY_SOCKET_PATH;

    // Add listening port (no authentication for UNIX sockets)
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

    // Register the service
    builder.RegisterService(service_impl_.get());

    // Assemble and start the server
    server_ = builder.BuildAndStart();

    if (!server_) {
        log_error(tt::LogAlways, "Failed to start gRPC server on UNIX socket: {}", GRPC_TELEMETRY_SOCKET_PATH);
        return;
    }

    log_info(tt::LogAlways, "gRPC telemetry server listening on UNIX socket: {}", GRPC_TELEMETRY_SOCKET_PATH);

    // Set permissions so other processes can access it
    // 0666 = read/write for owner, group, and others
    if (chmod(GRPC_TELEMETRY_SOCKET_PATH, 0666) != 0) {
        log_warning(tt::LogAlways, "Failed to set permissions on socket: {}", GRPC_TELEMETRY_SOCKET_PATH);
    }
}

void GrpcTelemetryServer::stop() {
    if (server_) {
        log_info(tt::LogAlways, "Shutting down gRPC telemetry server");

        // Shutdown the server with a deadline
        server_->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(5));

        // Wait for all pending RPCs to complete
        server_->Wait();

        server_.reset();

        // Clean up socket file
        unlink(GRPC_TELEMETRY_SOCKET_PATH);
    }
}

void GrpcTelemetryServer::on_telemetry_updated(const std::shared_ptr<TelemetrySnapshot>& delta) {
    std::lock_guard<std::mutex> lock(state_mutex_);  // use state mutex to protect metric_paths_by_name_
    update_name_to_paths_mapping(&metric_paths_by_name_, *delta);
}
