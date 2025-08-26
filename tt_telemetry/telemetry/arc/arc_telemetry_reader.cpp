#include <telemetry/arc/arc_telemetry_reader.hpp>

#include <tt-metalium/assert.hpp>
#include <telemetry/ethernet/ethernet_endpoint.hpp>
#include <tt-metalium/cluster.hpp>
#include <tt-logger/tt-logger.hpp>
#include <memory>
#include <vector>
#include <map>

/**************************************************************************************************
 ARCTelemetryReader Class
**************************************************************************************************/

ARCTelemetryReader::ARCTelemetryReader(ChipIdentifier chip_id, std::unique_ptr<tt::umd::TTDevice> device) :
    id(chip_id), device_(std::move(device)) {
    TT_FATAL(device_ != nullptr, "TTDevice cannot be null for chip {}", id);

    telemetry_reader_ = device_->get_arc_telemetry_reader();
    // Note: telemetry_reader_ may be null if ARC telemetry is not available
    // Use is_valid() to check before reading
}

uint32_t ARCTelemetryReader::read_value(tt::umd::wormhole::TelemetryTag tag) const {
    TT_ASSERT(
        device_->get_arch() == tt::ARCH::WORMHOLE_B0,
        "Attempting to read Wormhole telemetry tag on non-Wormhole chip {}",
        id);
    
    if (!is_valid()) {
        log_error(tt::LogAlways, "Cannot read telemetry value: ARC telemetry reader is not available for chip {}", id);
        return 0;
    }
    
    return telemetry_reader_->read_entry(tag);
}

uint32_t ARCTelemetryReader::read_value(tt::umd::blackhole::TelemetryTag tag) const {
    TT_ASSERT(
        device_->get_arch() == tt::ARCH::BLACKHOLE,
        "Attempting to read Blackhole telemetry tag on non-Blackhole chip {}",
        id);
    
    if (!is_valid()) {
        log_error(tt::LogAlways, "Cannot read telemetry value: ARC telemetry reader is not available for chip {}", id);
        return 0;
    }
    
    return telemetry_reader_->read_entry(tag);
}

tt::ARCH ARCTelemetryReader::get_arch() const { return device_->get_arch(); }

bool ARCTelemetryReader::is_valid() const {
    return telemetry_reader_ != nullptr;
}

/**************************************************************************************************
 Utility Functions
**************************************************************************************************/

std::map<ChipIdentifier, std::shared_ptr<ARCTelemetryReader>> create_arc_telemetry_readers_for_mmio_chips(
    const tt::Cluster& cluster) {
    std::map<ChipIdentifier, std::shared_ptr<ARCTelemetryReader>> arc_readers;

    // Get all chips using get_ethernet_endpoints_by_chip
    auto ethernet_endpoints_by_chip = get_ethernet_endpoints_by_chip(cluster);

    // Get the mapping from chip ID to PCI device number for MMIO-capable chips
    std::unordered_map<tt::umd::chip_id_t, tt::umd::chip_id_t> chips_with_mmio =
        cluster.get_cluster_desc()->get_chips_with_mmio();

    // Iterate through all chips and create ARCTelemetryReader instances for MMIO-capable ones
    for (const auto& [chip_identifier, endpoints] : ethernet_endpoints_by_chip) {
        tt::umd::chip_id_t chip_id = chip_identifier.id;

        // Check if this chip has MMIO capability (is a local chip)
        if (chips_with_mmio.find(chip_id) != chips_with_mmio.end()) {
            // This is a local chip - create TTDevice from PCI device number
            int pci_device_number = chips_with_mmio.at(chip_id);
            std::unique_ptr<tt::umd::TTDevice> tt_device = tt::umd::TTDevice::create(pci_device_number);

            if (tt_device) {
                // Initialize the device
                tt_device->init_tt_device();

                // Create ARCTelemetryReader with the TTDevice
                auto arc_reader = std::make_shared<ARCTelemetryReader>(chip_identifier, std::move(tt_device));
                arc_readers[chip_identifier] = arc_reader;
            }
        }
    }

    return arc_readers;
}
