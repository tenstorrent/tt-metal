// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/assert.hpp>

#include <telemetry/arc/arc_telemetry_reader.hpp>
#include <telemetry/ethernet/ethernet_endpoint.hpp>

#include <memory>
#include <vector>
#include <map>

/**************************************************************************************************
 ARCTelemetryReader Class
**************************************************************************************************/

ARCTelemetryReader::ARCTelemetryReader(ChipIdentifier chip_id, tt::umd::TTDevice* device) :
    id(chip_id), device_(device) {
    TT_FATAL(device_ != nullptr, "TTDevice cannot be null for chip {}", id);

    telemetry_reader_ = device_->get_arc_telemetry_reader();
    // Note: telemetry_reader_ may be null if ARC telemetry is not available
    // Use is_valid() to check before reading
}

uint32_t ARCTelemetryReader::read_value(tt::umd::TelemetryTag tag) const {
    if (!is_valid()) {
        log_error(tt::LogAlways, "Cannot read telemetry value: ARC telemetry reader is not available for chip {}", id);
        return 0;
    }

    // For Wormhole architecture, translate telemetry tags to wormhole-specific equivalents
    if (get_arch() == tt::ARCH::WORMHOLE_B0) {
        tt::umd::wormhole::TelemetryTag wormhole_tag = translate_to_wormhole_tag(tag);
        if (!telemetry_reader_->is_entry_available(wormhole_tag)) {
            log_error(tt::LogAlways, "Cannot read telemetry value: entry {} not available for chip {}", static_cast<unsigned>(wormhole_tag), id);
            return 0;
        }
        return telemetry_reader_->read_entry(static_cast<uint8_t>(wormhole_tag));
    }

    if (!telemetry_reader_->is_entry_available(tag)) {
        log_error(tt::LogAlways, "Cannot read telemetry value: entry {} not available for chip {}", static_cast<unsigned>(tag), id);
        return 0;
    }
    return telemetry_reader_->read_entry(tag);
}

tt::ARCH ARCTelemetryReader::get_arch() const { return device_->get_arch(); }

bool ARCTelemetryReader::is_valid() const {
    return telemetry_reader_ != nullptr;
}

tt::umd::wormhole::TelemetryTag ARCTelemetryReader::translate_to_wormhole_tag(tt::umd::TelemetryTag tag) const {
    switch (tag) {
        case tt::umd::TelemetryTag::AICLK:
            return tt::umd::wormhole::TelemetryTag::AICLK;
        case tt::umd::TelemetryTag::AXICLK:
            return tt::umd::wormhole::TelemetryTag::AXICLK;
        case tt::umd::TelemetryTag::ARCCLK:
            return tt::umd::wormhole::TelemetryTag::ARCCLK;
        case tt::umd::TelemetryTag::FAN_SPEED:
            return tt::umd::wormhole::TelemetryTag::FAN_SPEED;
        case tt::umd::TelemetryTag::TDP:
            return tt::umd::wormhole::TelemetryTag::TDP;
        case tt::umd::TelemetryTag::TDC:
            return tt::umd::wormhole::TelemetryTag::TDC;
        case tt::umd::TelemetryTag::VCORE:
            return tt::umd::wormhole::TelemetryTag::VCORE;
        case tt::umd::TelemetryTag::ASIC_TEMPERATURE:
            return tt::umd::wormhole::TelemetryTag::ASIC_TEMPERATURE;
        case tt::umd::TelemetryTag::BOARD_TEMPERATURE:
            return tt::umd::wormhole::TelemetryTag::BOARD_TEMPERATURE;
        default:
            TT_FATAL(false, "Unsupported telemetry tag {} for Wormhole architecture on chip {}", static_cast<int>(tag), id);
            return tt::umd::wormhole::TelemetryTag::NUMBER_OF_TAGS; // This line will never be reached due to TT_FATAL
    }
}

/**************************************************************************************************
 Utility Functions
**************************************************************************************************/

std::map<ChipIdentifier, std::shared_ptr<ARCTelemetryReader>> create_arc_telemetry_readers_for_mmio_chips(
    const std::unique_ptr<tt::umd::Cluster>& cluster) {
    tt::umd::tt_ClusterDescriptor* cluster_descriptor = cluster->get_cluster_description();
    std::map<ChipIdentifier, std::shared_ptr<ARCTelemetryReader>> arc_readers;

    // Get all chips using get_ethernet_endpoints_by_chip
    auto ethernet_endpoints_by_chip = get_ethernet_endpoints_by_chip(cluster);

    // Iterate through all chips and create ARCTelemetryReader instances for MMIO-capable ones
    for (const auto& [chip_identifier, endpoints] : ethernet_endpoints_by_chip) {
        tt::umd::chip_id_t chip_id = chip_identifier.id;

        // Check if this chip has MMIO capability (is a local chip)
        if (cluster_descriptor->is_chip_mmio_capable(chip_id)) {
            tt::umd::TTDevice* device = cluster->get_tt_device(chip_id);
            if (device) {
                // Create ARCTelemetryReader with the TTDevice
                auto arc_reader = std::make_shared<ARCTelemetryReader>(chip_identifier, device);
                arc_readers[chip_identifier] = arc_reader;
            }
        }
    }

    return arc_readers;
}
