// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <telemetry/arc/caching_arc_telemetry_reader.hpp>

#include <string_view>
#include <typeinfo>

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
        // Helper lambda to safely convert version objects to strings
        auto safe_version_to_string = [](auto&& version_getter, std::string_view name) -> std::optional<std::string> {
            try {
                return version_getter().to_string();
            } catch (const std::exception& e) {
                log_debug(tt::LogAlways, "Failed to convert {} to string ({}): {}", name, typeid(e).name(), e.what());
                return std::nullopt;
            }
        };

        // Helper lambda for optional version conversion
        auto safe_optional_version_to_string = [&](auto version_opt,
                                                   std::string_view name) -> std::optional<std::string> {
            if (version_opt) {
                return safe_version_to_string([&]() { return *version_opt; }, name);
            }
            return std::nullopt;
        };

        // Read firmware versions and convert to strings
        snapshot.firmware_version =
            safe_version_to_string([&]() { return firmware_provider_->get_firmware_version(); }, "firmware_version");
        snapshot.eth_fw_version =
            safe_optional_version_to_string(firmware_provider_->get_eth_fw_version_semver(), "eth_fw_version");
        snapshot.gddr_fw_version =
            safe_optional_version_to_string(firmware_provider_->get_gddr_fw_version(), "gddr_fw_version");
        snapshot.cm_fw_version =
            safe_optional_version_to_string(firmware_provider_->get_cm_fw_version(), "cm_fw_version");
        snapshot.dm_app_fw_version =
            safe_optional_version_to_string(firmware_provider_->get_dm_app_fw_version(), "dm_app_fw_version");
        snapshot.dm_bl_fw_version =
            safe_optional_version_to_string(firmware_provider_->get_dm_bl_fw_version(), "dm_bl_fw_version");
        snapshot.tt_flash_version =
            safe_optional_version_to_string(firmware_provider_->get_tt_flash_version(), "tt_flash_version");

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

    } catch (const std::runtime_error& e) {
        log_warning(
            tt::LogAlways,
            "Failed to read ARC telemetry (runtime_error): {}. "
            "Device may be busy or unavailable.",
            e.what());

        // Return default-constructed snapshot on error
        return ARCTelemetrySnapshot();

    } catch (const std::exception& e) {
        log_warning(
            tt::LogAlways,
            "Failed to read ARC telemetry ({}): {}. "
            "Unexpected error.",
            typeid(e).name(),
            e.what());

        // Return default-constructed snapshot on error
        return ARCTelemetrySnapshot();

    } catch (...) {
        log_warning(tt::LogAlways, "Failed to read ARC telemetry: Unknown exception type. Unexpected error.");
        return ARCTelemetrySnapshot();
    }
}
