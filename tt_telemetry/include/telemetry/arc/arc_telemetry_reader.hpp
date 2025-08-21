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
#include <third_party/umd/device/api/umd/device/tt_device/tt_device.h>
#include <third_party/umd/device/api/umd/device/types/wormhole_telemetry.h>
#include <third_party/umd/device/api/umd/device/types/blackhole_telemetry.h>
#include <tt-metalium/cluster.hpp>

class ARCTelemetryReader {
public:
    const ChipIdentifier id;

    // Constructor takes ownership of TTDevice
    explicit ARCTelemetryReader(ChipIdentifier chip_id, std::unique_ptr<tt::umd::TTDevice> device);

    // Read telemetry value for Wormhole chips
    uint32_t read_value(tt::umd::wormhole::TelemetryTag tag) const;

    // Read telemetry value for Blackhole chips
    uint32_t read_value(tt::umd::blackhole::TelemetryTag tag) const;

    // Get the chip architecture
    tt::ARCH get_arch() const;

private:
    std::unique_ptr<tt::umd::TTDevice> device_;
    tt::umd::ArcTelemetryReader* telemetry_reader_;
};

// Utility function to create ARC telemetry readers for all MMIO-capable chips
std::map<ChipIdentifier, std::shared_ptr<ARCTelemetryReader>> create_arc_telemetry_readers_for_mmio_chips(
    const tt::Cluster& cluster);
