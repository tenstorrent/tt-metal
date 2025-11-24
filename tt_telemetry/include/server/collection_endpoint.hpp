#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <future>
#include <string>
#include <utility>

#include <telemetry/telemetry_subscriber.hpp>

/*
 * server/collection_endpoint.hpp
 *
 * Collection endpoint server for broadcasting telemetry data using websocketpp.
 */

std::pair<std::future<bool>, std::shared_ptr<TelemetrySubscriber>> run_collection_endpoint(
    uint16_t port, const std::string& metal_home = "");
