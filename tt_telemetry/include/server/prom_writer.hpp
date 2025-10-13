#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <future>
#include <string>
#include <string_view>
#include <utility>

#include <telemetry/telemetry_subscriber.hpp>

/*
 * server/prom_writer.hpp
 *
 * File writer of telemetry data in Prometheus (.prom) format.
 */

std::pair<std::future<bool>, std::shared_ptr<TelemetrySubscriber>> run_prom_writer(std::string_view file_path);
