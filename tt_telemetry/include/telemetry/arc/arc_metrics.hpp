#pragma once

/*
 * telemetry/arc/arc_metrics.hpp
 *
 * ARC telemetry metrics that read from ARC (Argonaut RISC Core) processors on chips.
 * Includes both integral (uint) and floating-point (double) valued metrics.
 */

#include <memory>
#include <string>

#include <telemetry/metric.hpp>
#include <telemetry/arc/arc_telemetry_reader.hpp>
#include <third_party/umd/device/api/umd/device/types/wormhole_telemetry.h>
#include <third_party/umd/device/api/umd/device/types/blackhole_telemetry.h>

class ARCTelemetryAvailableMetric : public BoolMetric {
public:
    // Constructor
    ARCTelemetryAvailableMetric(size_t chip_id, std::shared_ptr<ARCTelemetryReader> reader);

    const std::vector<std::string> telemetry_path() const override;
    void update(const tt::Cluster& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    std::shared_ptr<ARCTelemetryReader> reader_;
};

class ARCUintMetric : public UIntMetric {
public:
    // Common integral metrics that exist on both architectures
    enum class CommonTelemetryTag {
        AICLK,      // AI clock frequency (MHz)
        AXICLK,     // AXI clock frequency (MHz)
        ARCCLK,     // ARC clock frequency (MHz)
        FAN_SPEED,  // Fan speed (RPM)
        TDP,        // Thermal Design Power (W)
        TDC,        // Thermal Design Current (A)
        VCORE       // Core voltage (mV)
    };

    // Constructor for Wormhole telemetry tags
    ARCUintMetric(
        size_t chip_id,
        std::shared_ptr<ARCTelemetryReader> reader,
        tt::umd::wormhole::TelemetryTag tag,
        const std::string& metric_name,
        uint32_t mask = 0xffffffff,
        MetricUnit units = MetricUnit::UNITLESS);

    // Constructor for Blackhole telemetry tags
    ARCUintMetric(
        size_t chip_id,
        std::shared_ptr<ARCTelemetryReader> reader,
        tt::umd::blackhole::TelemetryTag tag,
        const std::string& metric_name,
        uint32_t mask = 0xffffffff,
        MetricUnit units = MetricUnit::UNITLESS);

    // Constructor for common metrics (automatically selects appropriate tag based on architecture)
    ARCUintMetric(size_t chip_id, std::shared_ptr<ARCTelemetryReader> reader, CommonTelemetryTag common_metric);

    const std::vector<std::string> telemetry_path() const override;
    void update(const tt::Cluster& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    std::shared_ptr<ARCTelemetryReader> reader_;
    tt::umd::wormhole::TelemetryTag wormhole_tag_;
    tt::umd::blackhole::TelemetryTag blackhole_tag_;
    std::string metric_name_;
    uint32_t mask_;
};

class ARCDoubleMetric : public DoubleMetric {
public:
    // Common double-valued metrics that exist on both architectures
    enum class CommonTelemetryTag {
        ASIC_TEMPERATURE,  // ASIC temperature (Celsius, converted from raw)
        BOARD_TEMPERATURE  // Board temperature (Celsius, converted from raw)
    };

    // Constructor for Wormhole telemetry tags
    ARCDoubleMetric(
        size_t chip_id,
        std::shared_ptr<ARCTelemetryReader> reader,
        tt::umd::wormhole::TelemetryTag tag,
        const std::string& metric_name,
        uint32_t mask = 0xffffffff,
        double scale_factor = 1.0,
        MetricUnit units = MetricUnit::UNITLESS);

    // Constructor for Blackhole telemetry tags
    ARCDoubleMetric(
        size_t chip_id,
        std::shared_ptr<ARCTelemetryReader> reader,
        tt::umd::blackhole::TelemetryTag tag,
        const std::string& metric_name,
        uint32_t mask = 0xffffffff,
        double scale_factor = 1.0,
        MetricUnit units = MetricUnit::UNITLESS);

    // Constructor for common metrics (automatically selects appropriate tag based on architecture)
    ARCDoubleMetric(size_t chip_id, std::shared_ptr<ARCTelemetryReader> reader, CommonTelemetryTag common_metric);

    const std::vector<std::string> telemetry_path() const override;
    void update(const tt::Cluster& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    std::shared_ptr<ARCTelemetryReader> reader_;
    tt::umd::wormhole::TelemetryTag wormhole_tag_;
    tt::umd::blackhole::TelemetryTag blackhole_tag_;
    std::string metric_name_;
    uint32_t mask_;        // Mask to apply to raw telemetry value
    double scale_factor_;  // Factor to scale raw telemetry value to double
};
