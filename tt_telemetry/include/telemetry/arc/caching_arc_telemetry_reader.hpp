#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/arc/caching_arc_telemetry_reader.hpp
 *
 * Caching wrapper for ARC telemetry data from FirmwareInfoProvider.
 * Reads firmware telemetry once per update cycle and caches results to avoid
 * redundant reads when multiple metrics need the same data.
 */

#include <chrono>
#include <memory>
#include <cstdint>

#include <umd/device/firmware/firmware_info_provider.hpp>

#include <telemetry/arc/arc_telemetry_snapshot.hpp>
#include <telemetry/caching_telemetry_reader.hpp>

class CachingARCTelemetryReader final : public CachingTelemetryReader<ARCTelemetrySnapshot> {
public:
    // Constructor takes FirmwareInfoProvider reference
    // Note: firmware_provider must outlive this object (guaranteed by telemetry collector lifecycle)
    CachingARCTelemetryReader(tt::umd::FirmwareInfoProvider* firmware_provider);

    CachingARCTelemetryReader(const CachingARCTelemetryReader&) = delete;
    CachingARCTelemetryReader& operator=(const CachingARCTelemetryReader&) = delete;
    CachingARCTelemetryReader(CachingARCTelemetryReader&&) = delete;
    CachingARCTelemetryReader& operator=(CachingARCTelemetryReader&&) = delete;

private:
    ARCTelemetrySnapshot read_telemetry() override;

    tt::umd::FirmwareInfoProvider* firmware_provider_;
};
