// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <telemetry/arc/caching_arc_telemetry_reader.hpp>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

CachingARCTelemetryReader::CachingARCTelemetryReader(tt::umd::FirmwareInfoProvider* firmware_provider) :
    firmware_provider_(firmware_provider) {
    TT_FATAL(firmware_provider_ != nullptr, "CachingARCTelemetryReader: firmware_provider cannot be null");
    log_debug(tt::LogAlways, "CachingARCTelemetryReader initialized");
}

ARCTelemetrySnapshot CachingARCTelemetryReader::read_telemetry() {
    ARCTelemetrySnapshot snapshot;

    try {
        // Read firmware versions and convert to strings
        try {
            snapshot.firmware_version = firmware_provider_->get_firmware_version().to_string();
        } catch (...) {
            snapshot.firmware_version = std::nullopt;
        }

        if (auto version_opt = firmware_provider_->get_eth_fw_version_semver()) {
            try {
                snapshot.eth_fw_version = version_opt->to_string();
            } catch (...) {
                snapshot.eth_fw_version = std::nullopt;
            }
        }

        if (auto version_opt = firmware_provider_->get_gddr_fw_version()) {
            try {
                snapshot.gddr_fw_version = version_opt->to_string();
            } catch (...) {
                snapshot.gddr_fw_version = std::nullopt;
            }
        }

        if (auto version_opt = firmware_provider_->get_cm_fw_version()) {
            try {
                snapshot.cm_fw_version = version_opt->to_string();
            } catch (...) {
                snapshot.cm_fw_version = std::nullopt;
            }
        }

        if (auto version_opt = firmware_provider_->get_dm_app_fw_version()) {
            try {
                snapshot.dm_app_fw_version = version_opt->to_string();
            } catch (...) {
                snapshot.dm_app_fw_version = std::nullopt;
            }
        }

        if (auto version_opt = firmware_provider_->get_dm_bl_fw_version()) {
            try {
                snapshot.dm_bl_fw_version = version_opt->to_string();
            } catch (...) {
                snapshot.dm_bl_fw_version = std::nullopt;
            }
        }

        if (auto version_opt = firmware_provider_->get_tt_flash_version()) {
            try {
                snapshot.tt_flash_version = version_opt->to_string();
            } catch (...) {
                snapshot.tt_flash_version = std::nullopt;
            }
        }

        // Read board identification
        snapshot.board_id = firmware_provider_->get_board_id();
        snapshot.asic_location = firmware_provider_->get_asic_location();

        // Read temperature sensors
        snapshot.asic_temperature = firmware_provider_->get_asic_temperature();
        snapshot.board_temperature = firmware_provider_->get_board_temperature();

        // Read clock frequencies
        snapshot.aiclk = firmware_provider_->get_aiclk();
        snapshot.axiclk = firmware_provider_->get_axiclk();
        snapshot.arcclk = firmware_provider_->get_arcclk();
        snapshot.max_clock_freq = firmware_provider_->get_max_clock_freq();

        // Read power and cooling metrics
        snapshot.fan_speed = firmware_provider_->get_fan_speed();
        snapshot.tdp = firmware_provider_->get_tdp();
        snapshot.tdc = firmware_provider_->get_tdc();
        snapshot.vcore = firmware_provider_->get_vcore();

        // Read health monitoring
        snapshot.heartbeat = firmware_provider_->get_heartbeat();

        return snapshot;

    } catch (const std::exception& e) {
        log_warning(
            tt::LogAlways,
            "Failed to read ARC telemetry: {}. "
            "Device may be busy or unavailable.",
            e.what());

        // Return default-constructed snapshot on error
        return ARCTelemetrySnapshot();
    }
}
