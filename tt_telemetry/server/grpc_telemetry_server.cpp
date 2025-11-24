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

namespace tt {
namespace telemetry {

// Service implementation
class TelemetryServiceImpl final : public TelemetryService::Service {
public:
    TelemetryServiceImpl(const TelemetrySnapshot& telemetry_state, std::mutex& state_mutex) :
        telemetry_state_(telemetry_state), state_mutex_(state_mutex) {}
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

        log_debug(tt::LogAlways, "gRPC QueryMetric received for query: {}", metric_query);

        bool found_any = false;

        // Lock the mutex to safely access telemetry_state_
        std::lock_guard<std::mutex> lock(state_mutex_);

        // Iterate through all bool metrics and check if metric_query is a substring of the path
        for (const auto& [path, value] : telemetry_state_.bool_metrics) {
            if (path.find(metric_query) != std::string::npos) {
                auto* result = response->add_bool_results();
                result->set_path(path);
                result->set_value(value);

                // Get timestamp if available
                auto ts_it = telemetry_state_.bool_metric_timestamps.find(path);
                if (ts_it != telemetry_state_.bool_metric_timestamps.end()) {
                    result->set_timestamp(ts_it->second);
                } else {
                    result->set_timestamp(0);
                }

                log_debug(tt::LogAlways, "Found bool metric '{}' with value: {}", path, value);
                found_any = true;
            }
        }

        // Iterate through all uint metrics
        for (const auto& [path, value] : telemetry_state_.uint_metrics) {
            if (path.find(metric_query) != std::string::npos) {
                auto* result = response->add_uint_results();
                result->set_path(path);
                result->set_value(value);

                // Get timestamp if available
                auto ts_it = telemetry_state_.uint_metric_timestamps.find(path);
                if (ts_it != telemetry_state_.uint_metric_timestamps.end()) {
                    result->set_timestamp(ts_it->second);
                } else {
                    result->set_timestamp(0);
                }

                log_debug(tt::LogAlways, "Found uint metric '{}' with value: {}", path, value);
                found_any = true;
            }
        }

        // Iterate through all double metrics
        for (const auto& [path, value] : telemetry_state_.double_metrics) {
            if (path.find(metric_query) != std::string::npos) {
                auto* result = response->add_double_results();
                result->set_path(path);
                result->set_value(value);

                // Get timestamp if available
                auto ts_it = telemetry_state_.double_metric_timestamps.find(path);
                if (ts_it != telemetry_state_.double_metric_timestamps.end()) {
                    result->set_timestamp(ts_it->second);
                } else {
                    result->set_timestamp(0);
                }

                log_debug(tt::LogAlways, "Found double metric '{}' with value: {}", path, value);
                found_any = true;
            }
        }

        // Iterate through all string metrics
        for (const auto& [path, value] : telemetry_state_.string_metrics) {
            if (path.find(metric_query) != std::string::npos) {
                auto* result = response->add_string_results();
                result->set_path(path);
                result->set_value(value);

                // Get timestamp if available
                auto ts_it = telemetry_state_.string_metric_timestamps.find(path);
                if (ts_it != telemetry_state_.string_metric_timestamps.end()) {
                    result->set_timestamp(ts_it->second);
                } else {
                    result->set_timestamp(0);
                }

                log_debug(tt::LogAlways, "Found string metric '{}' with value: {}", path, value);
                found_any = true;
            }
        }

        // No matching metrics found
        if (!found_any) {
            log_debug(tt::LogAlways, "No metrics found matching query: '{}'", metric_query);
            return Status(grpc::StatusCode::NOT_FOUND, "No metrics found matching query: '" + metric_query + "'");
        }

        return Status::OK;
    }

private:
    // References to the telemetry state (from GrpcTelemetryServer/TelemetrySubscriber)
    const TelemetrySnapshot& telemetry_state_;
    std::mutex& state_mutex_;
};

}  // namespace telemetry
}  // namespace tt

GrpcTelemetryServer::GrpcTelemetryServer() :
    TelemetrySubscriber(),
    service_impl_(std::make_unique<tt::telemetry::TelemetryServiceImpl>(telemetry_state_, state_mutex_)) {}

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
    // Nothing to do here - QueryMetric will iterate through telemetry_state_ directly
}
