// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * ARC telemetry metrics using CachingARCTelemetryReader for cached access to telemetry data.
 */

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <telemetry/arc/arc_metrics.hpp>
#include <topology/topology.hpp>

#include <chrono>

/**************************************************************************************************
| Metric Creation
**************************************************************************************************/

// Creates caching ARC telemetry readers for each MMIO-capable (i.e., local) chip
std::unordered_map<tt::ChipId, std::shared_ptr<CachingARCTelemetryReader>> create_arc_telemetry_readers(
    const std::unique_ptr<tt::umd::Cluster>& cluster) {
    std::unordered_map<tt::ChipId, std::shared_ptr<CachingARCTelemetryReader>> telemetry_reader_by_chip_id;

    log_info(tt::LogAlways, "Creating ARC telemetry readers...");

    tt::umd::ClusterDescriptor* cluster_descriptor = cluster->get_cluster_description();
    for (tt::ChipId chip_id : cluster_descriptor->get_all_chips()) {
        if (cluster_descriptor->is_chip_mmio_capable(chip_id)) {
            tt::umd::TTDevice* device = cluster->get_tt_device(chip_id);
            if (device) {
                auto firmware_provider = device->get_firmware_info_provider();
                if (firmware_provider) {
                    std::shared_ptr<CachingARCTelemetryReader> telemetry_reader =
                        std::make_shared<CachingARCTelemetryReader>(firmware_provider);
                    telemetry_reader_by_chip_id[chip_id] = telemetry_reader;
                } else {
                    log_error(
                        tt::LogAlways,
                        "Unable to create ARC telemetry reader for chip_id={} because firmware provider does not exist",
                        chip_id);
                }
            } else {
                log_error(
                    tt::LogAlways,
                    "Unable to create ARC telemetry reader for chip_id={} because device is not accessible",
                    chip_id);
            }
        }
    }

    return telemetry_reader_by_chip_id;
}

// Creates ARC telemetry metrics for MMIO-capable chips
void create_arc_metrics(
    std::vector<std::unique_ptr<BoolMetric>>& bool_metrics,
    std::vector<std::unique_ptr<UIntMetric>>& uint_metrics,
    std::vector<std::unique_ptr<DoubleMetric>>& double_metrics,
    std::vector<std::unique_ptr<StringMetric>>& string_metrics,
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const std::unique_ptr<TopologyHelper>& topology_translation,
    const std::unique_ptr<tt::tt_metal::Hal>& hal,
    const std::unordered_map<tt::ChipId, std::shared_ptr<CachingARCTelemetryReader>>& telemetry_reader_by_chip_id) {
    log_info(tt::LogAlways, "Creating ARC firmware metrics...");

    // Iterate through all chips and create ARC metrics for MMIO-capable ones for which we have a telemetry reader
    for (const auto& [chip_id, telemetry_reader] : telemetry_reader_by_chip_id) {
        // Get ASICDescriptor
        std::optional<tt::tt_metal::ASICDescriptor> asic_descriptor =
            topology_translation->get_asic_descriptor_for_local_chip(chip_id);
        TT_FATAL(asic_descriptor.has_value(), "No ASIC descriptor for chip ID {}", chip_id);

        log_info(
            tt::LogAlways,
            "Creating ARC firmware metrics for tray_id={}, asic_location={}, chip_id={}...",
            *asic_descriptor.value().tray_id,
            *asic_descriptor.value().asic_location,
            chip_id);

        // Create integer metrics
        uint_metrics.push_back(std::make_unique<ARCUintMetric>(
            asic_descriptor.value(),
            telemetry_reader,
            "AIClock",
            [](const ARCTelemetrySnapshot* snapshot) { return snapshot->aiclk; },
            MetricUnit::MEGAHERTZ));

        uint_metrics.push_back(std::make_unique<ARCUintMetric>(
            asic_descriptor.value(),
            telemetry_reader,
            "AXIClock",
            [](const ARCTelemetrySnapshot* snapshot) { return snapshot->axiclk; },
            MetricUnit::MEGAHERTZ));

        uint_metrics.push_back(std::make_unique<ARCUintMetric>(
            asic_descriptor.value(),
            telemetry_reader,
            "ARCClock",
            [](const ARCTelemetrySnapshot* snapshot) { return snapshot->arcclk; },
            MetricUnit::MEGAHERTZ));

        uint_metrics.push_back(std::make_unique<ARCUintMetric>(
            asic_descriptor.value(),
            telemetry_reader,
            "FanSpeed",
            [](const ARCTelemetrySnapshot* snapshot) { return snapshot->fan_speed; },
            MetricUnit::REVOLUTIONS_PER_MINUTE));

        uint_metrics.push_back(std::make_unique<ARCUintMetric>(
            asic_descriptor.value(),
            telemetry_reader,
            "TDP",
            [](const ARCTelemetrySnapshot* snapshot) { return snapshot->tdp; },
            MetricUnit::WATTS));

        uint_metrics.push_back(std::make_unique<ARCUintMetric>(
            asic_descriptor.value(),
            telemetry_reader,
            "TDC",
            [](const ARCTelemetrySnapshot* snapshot) { return snapshot->tdc; },
            MetricUnit::AMPERES));

        uint_metrics.push_back(std::make_unique<ARCUintMetric>(
            asic_descriptor.value(),
            telemetry_reader,
            "VCore",
            [](const ARCTelemetrySnapshot* snapshot) { return snapshot->vcore; },
            MetricUnit::MILLIVOLTS));

        // Create double metrics
        double_metrics.push_back(std::make_unique<ARCDoubleMetric>(
            asic_descriptor.value(),
            telemetry_reader,
            "ASICTemperature",
            [](const ARCTelemetrySnapshot* snapshot) { return std::optional<double>(snapshot->asic_temperature); },
            MetricUnit::CELSIUS));

        double_metrics.push_back(std::make_unique<ARCDoubleMetric>(
            asic_descriptor.value(),
            telemetry_reader,
            "BoardTemperature",
            [](const ARCTelemetrySnapshot* snapshot) { return snapshot->board_temperature; },
            MetricUnit::CELSIUS));

        // Create String metrics for firmware version information
        string_metrics.push_back(std::make_unique<ARCStringMetric>(
            asic_descriptor.value(),
            telemetry_reader,
            "FirmwareBundleVersion",
            [](const ARCTelemetrySnapshot* snapshot) { return snapshot->firmware_version; },
            MetricUnit::UNITLESS));

        string_metrics.push_back(std::make_unique<ARCStringMetric>(
            asic_descriptor.value(),
            telemetry_reader,
            "EthernetFirmwareVersion",
            [](const ARCTelemetrySnapshot* snapshot) { return snapshot->eth_fw_version; },
            MetricUnit::UNITLESS));

        string_metrics.push_back(std::make_unique<ARCStringMetric>(
            asic_descriptor.value(),
            telemetry_reader,
            "CMFirmwareVersion",
            [](const ARCTelemetrySnapshot* snapshot) { return snapshot->cm_fw_version; },
            MetricUnit::UNITLESS));

        string_metrics.push_back(std::make_unique<ARCStringMetric>(
            asic_descriptor.value(),
            telemetry_reader,
            "DMAppFirmwareVersion",
            [](const ARCTelemetrySnapshot* snapshot) { return snapshot->dm_app_fw_version; },
            MetricUnit::UNITLESS));

        string_metrics.push_back(std::make_unique<ARCStringMetric>(
            asic_descriptor.value(),
            telemetry_reader,
            "DMBootloaderFirmwareVersion",
            [](const ARCTelemetrySnapshot* snapshot) { return snapshot->dm_bl_fw_version; },
            MetricUnit::UNITLESS));

        string_metrics.push_back(std::make_unique<ARCStringMetric>(
            asic_descriptor.value(),
            telemetry_reader,
            "TTFlashVersion",
            [](const ARCTelemetrySnapshot* snapshot) { return snapshot->tt_flash_version; },
            MetricUnit::UNITLESS));
    }

    log_info(tt::LogAlways, "Created ARC metrics");
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
    std::shared_ptr<CachingARCTelemetryReader> telemetry_reader,
    const std::string& metric_name,
    std::function<std::optional<uint32_t>(const ARCTelemetrySnapshot*)> getter_func,
    MetricUnit units) :
    UIntMetric(units),
    asic_descriptor_(asic_descriptor),
    telemetry_reader_(telemetry_reader),
    metric_name_(metric_name),
    getter_func_(getter_func) {
    TT_ASSERT(telemetry_reader_ != nullptr, "CachingARCTelemetryReader cannot be null");
    value_ = 0;
}

const std::vector<std::string> ARCUintMetric::telemetry_path() const {
    return arc_telemetry_path(asic_descriptor_.tray_id, asic_descriptor_.asic_location, metric_name_);
}

void ARCUintMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    // Get cached telemetry snapshot
    const ARCTelemetrySnapshot* snapshot = telemetry_reader_->get_telemetry(start_of_update_cycle);

    // Get the value using the getter function
    auto optional_value = getter_func_(snapshot);

    // Update the metric value and timestamp
    uint64_t new_value = optional_value.value_or(0);
    uint64_t old_value = value_;
    changed_since_transmission_ = new_value != old_value;
    value_ = new_value;
    set_timestamp_now();
}

/**************************************************************************************************
| ARCDoubleMetric Class
**************************************************************************************************/

ARCDoubleMetric::ARCDoubleMetric(
    tt::tt_metal::ASICDescriptor asic_descriptor,
    std::shared_ptr<CachingARCTelemetryReader> telemetry_reader,
    const std::string& metric_name,
    std::function<std::optional<double>(const ARCTelemetrySnapshot*)> getter_func,
    MetricUnit units) :
    DoubleMetric(units),
    asic_descriptor_(asic_descriptor),
    telemetry_reader_(telemetry_reader),
    metric_name_(metric_name),
    getter_func_(getter_func) {
    TT_ASSERT(telemetry_reader_ != nullptr, "CachingARCTelemetryReader cannot be null");
    value_ = 0.0;
}

const std::vector<std::string> ARCDoubleMetric::telemetry_path() const {
    return arc_telemetry_path(asic_descriptor_.tray_id, asic_descriptor_.asic_location, metric_name_);
}

void ARCDoubleMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    // Get cached telemetry snapshot
    const ARCTelemetrySnapshot* snapshot = telemetry_reader_->get_telemetry(start_of_update_cycle);

    // Get the value using the getter function
    auto optional_value = getter_func_(snapshot);

    // Update the metric value and timestamp
    double new_value = optional_value.value_or(0.0);
    double old_value = value_;
    changed_since_transmission_ = new_value != old_value;
    value_ = new_value;
    set_timestamp_now();
}

/**************************************************************************************************
| ARCStringMetric Class
**************************************************************************************************/

ARCStringMetric::ARCStringMetric(
    tt::tt_metal::ASICDescriptor asic_descriptor,
    std::shared_ptr<CachingARCTelemetryReader> telemetry_reader,
    const std::string& metric_name,
    std::function<std::optional<std::string>(const ARCTelemetrySnapshot*)> getter_func,
    MetricUnit units) :
    StringMetric(units),
    asic_descriptor_(asic_descriptor),
    telemetry_reader_(telemetry_reader),
    metric_name_(metric_name),
    getter_func_(getter_func) {
    TT_ASSERT(telemetry_reader_ != nullptr, "CachingARCTelemetryReader cannot be null");
    value_ = "";
}

const std::vector<std::string> ARCStringMetric::telemetry_path() const {
    return arc_telemetry_path(asic_descriptor_.tray_id, asic_descriptor_.asic_location, metric_name_);
}

void ARCStringMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    // Get cached telemetry snapshot
    const ARCTelemetrySnapshot* snapshot = telemetry_reader_->get_telemetry(start_of_update_cycle);

    // Get the value using the getter function
    auto optional_value = getter_func_(snapshot);

    // Update the metric value and timestamp
    std::string new_value = optional_value.value_or("");
    changed_since_transmission_ = (new_value != value_);
    value_ = std::move(new_value);
    set_timestamp_now();
}
