#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/arc/arc_telemetry_snapshot.hpp
 *
 * Container for all telemetry data available from FirmwareInfoProvider.
 * This struct captures all possible metrics that can be read from ARC firmware.
 */

#include <cstdint>
#include <optional>
#include <string>

/*
 * ARCTelemetrySnapshot
 *
 * Container for all telemetry data available from FirmwareInfoProvider.
 * This struct captures all possible metrics that can be read from ARC firmware.
 */
struct ARCTelemetrySnapshot {
    // Firmware versions (as strings)
    std::optional<std::string> firmware_version;
    std::optional<std::string> eth_fw_version;
    std::optional<std::string> gddr_fw_version;
    std::optional<std::string> cm_fw_version;
    std::optional<std::string> dm_app_fw_version;
    std::optional<std::string> dm_bl_fw_version;
    std::optional<std::string> tt_flash_version;

    // Board identification
    uint64_t board_id;
    uint8_t asic_location;

    // Temperature sensors (Celsius)
    double asic_temperature;                  // °C
    std::optional<double> board_temperature;  // °C

    // Clock frequencies (MHz)
    std::optional<uint32_t> aiclk;
    std::optional<uint32_t> axiclk;
    std::optional<uint32_t> arcclk;
    uint32_t max_clock_freq;

    // Power and cooling
    std::optional<uint32_t> fan_speed;  // RPM
    std::optional<uint32_t> tdp;        // Watts
    std::optional<uint32_t> tdc;        // Amps
    std::optional<uint32_t> vcore;      // Millivolts

    // Health monitoring
    uint32_t heartbeat;

    // Default constructor initializes with safe defaults
    ARCTelemetrySnapshot() : board_id(0), asic_location(0), asic_temperature(0.0), max_clock_freq(0), heartbeat(0) {}
};
