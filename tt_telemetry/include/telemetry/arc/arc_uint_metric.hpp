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

    const std::vector<std::string> telemetry_path() const override;
    void update(const tt::Cluster& cluster) override;

private:
    std::shared_ptr<ARCTelemetryReader> reader_;
    tt::umd::wormhole::TelemetryTag wormhole_tag_;
    tt::umd::blackhole::TelemetryTag blackhole_tag_;
    std::string metric_name_;
};
