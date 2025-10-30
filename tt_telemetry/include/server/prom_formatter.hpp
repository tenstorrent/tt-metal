// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <telemetry/telemetry_snapshot.hpp>

namespace tt::telemetry {

// Formats a TelemetrySnapshot into Prometheus exposition format
// This generates plaintext output suitable for Prometheus scraping.
// Hostname is extracted from each metric path and added as a label
std::string format_snapshot_as_prometheus(const TelemetrySnapshot& snapshot);

}  // namespace tt::telemetry
