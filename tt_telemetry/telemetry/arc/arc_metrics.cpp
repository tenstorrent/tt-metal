// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * ARC telemetry metrics using FirmwareInfoProvider for direct access to telemetry data.
 *
 * TODO:
 * -----
 * - How to handle cases where an ARC telemetry value is not returned by FirmwareInfoProvider?
 *   For now, we set it to 0. But maybe we want to stop updating it at all and retain the old
 *   stale value instead?
 */

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <telemetry/arc/arc_metrics.hpp>
#include <topology/topology.hpp>

#include <chrono>

/**************************************************************************************************
| Metric Creation
**************************************************************************************************/

// Creates ARC telemetry metrics for MMIO-capable chips using FirmwareInfoProvider
void create_arc_metrics(
    std::vector<std::unique_ptr<BoolMetric>>& bool_metrics,
    std::vector<std::unique_ptr<UIntMetric>>& uint_metrics,
    std::vector<std::unique_ptr<DoubleMetric>>& double_metrics,
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const std::unique_ptr<TopologyHelper>& topology_translation,
    const std::unique_ptr<tt::tt_metal::Hal>& hal) {
    tt::umd::tt_ClusterDescriptor* cluster_descriptor = cluster->get_cluster_description();

    // Iterate through all chips and create ARC metrics for MMIO-capable ones
    for (ChipId chip_id : cluster_descriptor->get_all_chips()) {
        // Check if this chip has MMIO capability (is a local chip)
        if (cluster_descriptor->is_chip_mmio_capable(chip_id)) {
            tt::umd::TTDevice* device = cluster->get_tt_device(chip_id);
            if (device) {
                // Get ASICDescriptor
                std::optional<tt::tt_metal::ASICDescriptor> asic_descriptor =
                    topology_translation->get_asic_descriptor_for_local_chip(chip_id);
                TT_FATAL(asic_descriptor.has_value(), "No ASIC descriptor for chip ID {}", chip_id);

                // Get FirmwareInfoProvider from the device
                auto firmware_provider = device->get_firmware_info_provider();
                if (firmware_provider) {
                    // Create UInt metrics using FirmwareInfoProvider methods
                    uint_metrics.push_back(std::make_unique<ARCUintMetric>(
                        asic_descriptor.value(),
                        firmware_provider,
                        "AIClock",
                        [firmware_provider]() { return firmware_provider->get_aiclk(); },
                        MetricUnit::MEGAHERTZ));

                    uint_metrics.push_back(std::make_unique<ARCUintMetric>(
                        asic_descriptor.value(),
                        firmware_provider,
                        "AXIClock",
                        [firmware_provider]() { return firmware_provider->get_axiclk(); },
                        MetricUnit::MEGAHERTZ));

                    uint_metrics.push_back(std::make_unique<ARCUintMetric>(
                        asic_descriptor.value(),
                        firmware_provider,
                        "ARCClock",
                        [firmware_provider]() { return firmware_provider->get_arcclk(); },
                        MetricUnit::MEGAHERTZ));

                    uint_metrics.push_back(std::make_unique<ARCUintMetric>(
                        asic_descriptor.value(),
                        firmware_provider,
                        "FanSpeed",
                        [firmware_provider]() { return firmware_provider->get_fan_speed(); },
                        MetricUnit::REVOLUTIONS_PER_MINUTE));

                    uint_metrics.push_back(std::make_unique<ARCUintMetric>(
                        asic_descriptor.value(),
                        firmware_provider,
                        "TDP",
                        [firmware_provider]() { return firmware_provider->get_tdp(); },
                        MetricUnit::WATTS));

                    uint_metrics.push_back(std::make_unique<ARCUintMetric>(
                        asic_descriptor.value(),
                        firmware_provider,
                        "TDC",
                        [firmware_provider]() { return firmware_provider->get_tdc(); },
                        MetricUnit::AMPERES));

                    uint_metrics.push_back(std::make_unique<ARCUintMetric>(
                        asic_descriptor.value(),
                        firmware_provider,
                        "VCore",
                        [firmware_provider]() { return firmware_provider->get_vcore(); },
                        MetricUnit::MILLIVOLTS));

                    // Create Double metrics using FirmwareInfoProvider methods
                    double_metrics.push_back(std::make_unique<ARCDoubleMetric>(
                        asic_descriptor.value(),
                        firmware_provider,
                        "ASICTemperature",
                        [firmware_provider]() {
                            return std::optional<double>(firmware_provider->get_asic_temperature());
                        },
                        MetricUnit::CELSIUS));

                    double_metrics.push_back(std::make_unique<ARCDoubleMetric>(
                        asic_descriptor.value(),
                        firmware_provider,
                        "BoardTemperature",
                        [firmware_provider]() { return firmware_provider->get_board_temperature(); },
                        MetricUnit::CELSIUS));
                }
            }
        }
    }
    log_info(tt::LogAlways, "Created ARC metrics using FirmwareInfoProvider");
}

static std::vector<std::string> arc_telemetry_path(
    tt::tt_metal::TrayID tray_id, tt::tt_metal::ASICLocation asic_location, const std::string& metric_name) {
    // Create path in format: tray{n}/chip{m}/metric_name
    return {"tray" + std::to_string(*tray_id), "chip" + std::to_string(*asic_location), metric_name};
}

/**************************************************************************************************
| ARCUintMetric Class
**************************************************************************************************/

ARCUintMetric::ARCUintMetric(
    tt::tt_metal::ASICDescriptor asic_descriptor,
    tt::umd::FirmwareInfoProvider* firmware_provider,
    const std::string& metric_name,
    std::function<std::optional<uint32_t>()> getter_func,
    MetricUnit units) :
    UIntMetric(units),
    asic_descriptor_(asic_descriptor),
    firmware_provider_(firmware_provider),
    metric_name_(metric_name),
    getter_func_(getter_func) {
    TT_ASSERT(firmware_provider_ != nullptr, "FirmwareInfoProvider cannot be null");
    value_ = 0;
}

const std::vector<std::string> ARCUintMetric::telemetry_path() const {
    return arc_telemetry_path(asic_descriptor_.tray_id, asic_descriptor_.asic_location, metric_name_);
}

void ARCUintMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    // Get the value using the getter function
    auto optional_value = getter_func_();

    // Update the metric value and timestamp
    uint64_t new_value = optional_value.value_or(0);
    uint64_t old_value = value_;
    changed_since_transmission_ = new_value != old_value;
    value_ = new_value;
    timestamp_ =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
}

/**************************************************************************************************
| ARCDoubleMetric Class
**************************************************************************************************/

ARCDoubleMetric::ARCDoubleMetric(
    tt::tt_metal::ASICDescriptor asic_descriptor,
    tt::umd::FirmwareInfoProvider* firmware_provider,
    const std::string& metric_name,
    std::function<std::optional<double>()> getter_func,
    MetricUnit units) :
    DoubleMetric(units),
    asic_descriptor_(asic_descriptor),
    firmware_provider_(firmware_provider),
    metric_name_(metric_name),
    getter_func_(getter_func) {
    TT_ASSERT(firmware_provider_ != nullptr, "FirmwareInfoProvider cannot be null");
    value_ = 0.0;
}

const std::vector<std::string> ARCDoubleMetric::telemetry_path() const {
    return arc_telemetry_path(asic_descriptor_.tray_id, asic_descriptor_.asic_location, metric_name_);
}

void ARCDoubleMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    // Get the value using the getter function
    auto optional_value = getter_func_();

    // Update the metric value and timestamp
    double new_value = optional_value.value_or(0.0);
    double old_value = value_;
    changed_since_transmission_ = new_value != old_value;
    value_ = new_value;
    timestamp_ =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
}
