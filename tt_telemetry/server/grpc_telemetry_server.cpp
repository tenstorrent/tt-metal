// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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
    TelemetryServiceImpl(TelemetrySnapshot& telemetry_state, std::mutex& state_mutex) :
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
        const std::string& metric_name = request->metric_name();

        log_debug(tt::LogAlways, "gRPC QueryMetric received for metric: {}", metric_name);

        // Lock the mutex to safely access telemetry_state_
        std::lock_guard<std::mutex> lock(state_mutex_);

        // Check each metric type map to find the requested metric
        // Try bool metrics first
        auto bool_it = telemetry_state_.bool_metrics.find(metric_name);
        if (bool_it != telemetry_state_.bool_metrics.end()) {
            response->set_metric_name(metric_name);
            response->set_bool_value(bool_it->second);
            log_debug(tt::LogAlways, "Found bool metric '{}' with value: {}", metric_name, bool_it->second);
            return Status::OK;
        }

        // Try uint metrics
        auto uint_it = telemetry_state_.uint_metrics.find(metric_name);
        if (uint_it != telemetry_state_.uint_metrics.end()) {
            response->set_metric_name(metric_name);
            response->set_uint_value(uint_it->second);
            log_debug(tt::LogAlways, "Found uint metric '{}' with value: {}", metric_name, uint_it->second);
            return Status::OK;
        }

        // Try double metrics
        auto double_it = telemetry_state_.double_metrics.find(metric_name);
        if (double_it != telemetry_state_.double_metrics.end()) {
            response->set_metric_name(metric_name);
            response->set_double_value(double_it->second);
            log_debug(tt::LogAlways, "Found double metric '{}' with value: {}", metric_name, double_it->second);
            return Status::OK;
        }

        // Try string metrics
        auto string_it = telemetry_state_.string_metrics.find(metric_name);
        if (string_it != telemetry_state_.string_metrics.end()) {
            response->set_metric_name(metric_name);
            response->set_string_value(string_it->second);
            log_debug(tt::LogAlways, "Found string metric '{}' with value: {}", metric_name, string_it->second);
            return Status::OK;
        }

        // Metric not found in any map - return error status
        log_debug(tt::LogAlways, "Metric '{}' not found in telemetry state", metric_name);
        return Status(grpc::StatusCode::NOT_FOUND, "Metric '" + metric_name + "' not found");
    }

private:
    // References to the telemetry state (from GrpcTelemetryServer/TelemetrySubscriber)
    TelemetrySnapshot& telemetry_state_;
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

void GrpcTelemetryServer::on_telemetry_updated(const TelemetrySnapshot& delta) {
    // For now, we don't need to do anything here
    // This will be used when we implement actual telemetry streaming/retrieval
    log_debug(tt::LogAlways, "gRPC server received telemetry update");
}
