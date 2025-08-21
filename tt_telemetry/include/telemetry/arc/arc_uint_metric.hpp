#pragma once

/*
 * telemetry/arc/arc_uint_metric.hpp
 *
 * ARC telemetry uint metric that reads from ARC (Argonaut RISC Core) processors on chips.
 */

#include <memory>
#include <string>

#include <telemetry/metric.hpp>
#include <telemetry/arc/arc_telemetry_reader.hpp>
#include <third_party/umd/device/api/umd/device/types/wormhole_telemetry.h>
#include <third_party/umd/device/api/umd/device/types/blackhole_telemetry.h>

class ARCUintMetric : public UIntMetric {
public:
    // Common integral metrics that exist on both architectures
    enum class CommonTelemetryTag {
        AICLK,      // AI clock frequency (MHz)
        AXICLK,     // AXI clock frequency (MHz)
        ARCCLK,     // ARC clock frequency (MHz)
        FAN_SPEED,  // Fan speed (RPM)
        TDP,        // Thermal Design Power
        TDC,        // Thermal Design Current
        VCORE       // Core voltage
    };

    // Constructor for Wormhole telemetry tags
    ARCUintMetric(
        size_t chip_id,
        std::shared_ptr<ARCTelemetryReader> reader,
        tt::umd::wormhole::TelemetryTag tag,
        const std::string& metric_name);

    // Constructor for Blackhole telemetry tags
    ARCUintMetric(
        size_t chip_id,
        std::shared_ptr<ARCTelemetryReader> reader,
        tt::umd::blackhole::TelemetryTag tag,
        const std::string& metric_name);

    // Constructor for common metrics (automatically selects appropriate tag based on architecture)
    ARCUintMetric(size_t chip_id, std::shared_ptr<ARCTelemetryReader> reader, CommonTelemetryTag common_metric);

    const std::vector<std::string> telemetry_path() const override;
    void update(const tt::Cluster& cluster) override;

private:
    std::shared_ptr<ARCTelemetryReader> reader_;
    tt::umd::wormhole::TelemetryTag wormhole_tag_;
    tt::umd::blackhole::TelemetryTag blackhole_tag_;
    std::string metric_name_;
};
