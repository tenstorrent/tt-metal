#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <future>
#include <string>
#include <utility>

#include <telemetry/telemetry_subscriber.hpp>

/*
 * server/websocket_server.hpp
 *
 * WebSocket server for broadcasting telemetry data using websocketpp.
 */

std::pair<std::future<bool>, std::shared_ptr<TelemetrySubscriber>> run_web_socket_server(
    uint16_t port, const std::string& metal_home = "");
