#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * server/grpc_telemetry_server.hpp
 *
 * gRPC server for telemetry service running on UNIX domain socket.
 * Implements TelemetrySubscriber to receive telemetry updates.
 */

#include <memory>
#include <string>
#include <thread>
#include <grpcpp/grpcpp.h>
#include <unordered_map>
#include <unordered_set>

#include <telemetry/telemetry_subscriber.hpp>

// Forward declaration - we use grpc::Service as the base type
namespace grpc {
class Service;
}

// UNIX domain socket path for the gRPC server
constexpr const char* GRPC_TELEMETRY_SOCKET_PATH = "/tmp/tt_telemetry2.sock";

class GrpcTelemetryServer : public TelemetrySubscriber {
public:
    GrpcTelemetryServer();
    ~GrpcTelemetryServer() override;

    // Start the gRPC server (non-blocking)
    void start();

    // Stop the gRPC server and wait for shutdown
    void stop();

protected:
    // Called when telemetry is updated
    void on_telemetry_updated(const TelemetrySnapshot& delta) override;

private:
    // gRPC server instance
    std::unique_ptr<grpc::Server> server_;

    // Service implementation instance (stored as base grpc::Service pointer)
    std::unique_ptr<grpc::Service> service_impl_;

    // Mapping of a metric name (the last part of a path) -> all possible paths containing that name
    std::unordered_map<std::string, std::unordered_set<std::string>> metric_paths_by_name_;
};
