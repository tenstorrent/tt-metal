#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <future>
#include <utility>

#include <telemetry/telemetry_subscriber.hpp>

/*
 * server/web_server.hpp
 *
 * Built-in web server for broadcasting telemetry data.
 */

std::pair<std::future<bool>, std::shared_ptr<TelemetrySubscriber>> run_web_server(uint16_t port);
