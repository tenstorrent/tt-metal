#pragma once

/*
 * telemetry/arc/arc_telemetry_reader.hpp
 *
 * ARC telemetry reader for Tenstorrent devices. Provides a unified interface
 * for reading telemetry data from ARC (Argonaut RISC Core) processors on chips.
 */

#include <memory>
#include <cstdint>
#include <vector>
#include <map>

#include <telemetry/ethernet/chip_identifier.hpp>
#include <third_party/umd/device/api/umd/device/cluster.hpp>
#include <third_party/umd/device/api/umd/device/types/telemetry.hpp>

class ARCTelemetryReader {
public:
    const ChipIdentifier id;

    // Constructor takes ownership of TTDevice
    explicit ARCTelemetryReader(ChipIdentifier chip_id, std::unique_ptr<tt::umd::TTDevice> device);

    // Read telemetry value using common telemetry tags
    uint32_t read_value(tt::umd::TelemetryTag tag) const;

    // Get the chip architecture
    tt::ARCH get_arch() const;

    // Check if the telemetry reader is valid (not null)
    bool is_valid() const;

private:
    std::unique_ptr<tt::umd::TTDevice> device_;
    tt::umd::ArcTelemetryReader* telemetry_reader_;
};

// Utility function to create ARC telemetry readers for all MMIO-capable chips
std::map<ChipIdentifier, std::shared_ptr<ARCTelemetryReader>> create_arc_telemetry_readers_for_mmio_chips(
    const std::unique_ptr<tt::umd::Cluster>& cluster);
