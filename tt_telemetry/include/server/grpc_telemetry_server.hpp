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

#include <telemetry/telemetry_subscriber.hpp>

// Forward declaration - we use grpc::Service as the base type
namespace grpc {
class Service;
}

class GrpcTelemetryServer : public TelemetrySubscriber {
public:
    explicit GrpcTelemetryServer(const std::string& socket_path);
    ~GrpcTelemetryServer() override;

    // Start the gRPC server (non-blocking)
    void start();

    // Stop the gRPC server and wait for shutdown
    void stop();

protected:
    // Called when telemetry is updated
    void on_telemetry_updated(const std::shared_ptr<TelemetrySnapshot>& delta) override;

private:
    // UNIX domain socket path
    std::string socket_path_;

    // gRPC server instance
    std::unique_ptr<grpc::Server> server_;

    // Service implementation instance (stored as base grpc::Service pointer)
    std::unique_ptr<grpc::Service> service_impl_;
};
