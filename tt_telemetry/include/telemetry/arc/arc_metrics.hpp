#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/arc/arc_metrics.hpp
 *
 * ARC telemetry metrics that read from ARC (Argonaut RISC Core) processors on chips.
 * Includes both integral (uint) and floating-point (double) valued metrics.
 */

#include <memory>
#include <string>
#include <functional>
#include <optional>
#include <unordered_map>

#include <llrt/hal.hpp>
#include <umd/device/cluster.hpp>

#include <telemetry/metric.hpp>
#include <telemetry/arc/caching_arc_telemetry_reader.hpp>
#include <tt_metal/fabric/physical_system_descriptor.hpp>

class TopologyHelper;

class ARCUintMetric : public UIntMetric {
public:
    // Constructor using CachingARCTelemetryReader
    ARCUintMetric(
        tt::tt_metal::ASICDescriptor asic_descriptor,
        std::shared_ptr<CachingARCTelemetryReader> telemetry_reader,
        const std::string& metric_name,
        std::function<std::optional<uint32_t>(const ARCTelemetrySnapshot*)> getter_func,
        MetricUnit units = MetricUnit::UNITLESS);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    tt::tt_metal::ASICDescriptor asic_descriptor_;
    std::shared_ptr<CachingARCTelemetryReader> telemetry_reader_;
    std::string metric_name_;
    std::function<std::optional<uint32_t>(const ARCTelemetrySnapshot*)> getter_func_;
};

class ARCDoubleMetric : public DoubleMetric {
public:
    // Constructor using CachingARCTelemetryReader
    ARCDoubleMetric(
        tt::tt_metal::ASICDescriptor asic_descriptor,
        std::shared_ptr<CachingARCTelemetryReader> telemetry_reader,
        const std::string& metric_name,
        std::function<std::optional<double>(const ARCTelemetrySnapshot*)> getter_func,
        MetricUnit units = MetricUnit::UNITLESS);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    tt::tt_metal::ASICDescriptor asic_descriptor_;
    std::shared_ptr<CachingARCTelemetryReader> telemetry_reader_;
    std::string metric_name_;
    std::function<std::optional<double>(const ARCTelemetrySnapshot*)> getter_func_;
};

class ARCStringMetric : public StringMetric {
public:
    // Constructor using CachingARCTelemetryReader
    ARCStringMetric(
        tt::tt_metal::ASICDescriptor asic_descriptor,
        std::shared_ptr<CachingARCTelemetryReader> telemetry_reader,
        const std::string& metric_name,
        std::function<std::optional<std::string>(const ARCTelemetrySnapshot*)> getter_func,
        MetricUnit units = MetricUnit::UNITLESS);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    tt::tt_metal::ASICDescriptor asic_descriptor_;
    std::shared_ptr<CachingARCTelemetryReader> telemetry_reader_;
    std::string metric_name_;
    std::function<std::optional<std::string>(const ARCTelemetrySnapshot*)> getter_func_;
};

std::unordered_map<tt::ChipId, std::shared_ptr<CachingARCTelemetryReader>> create_arc_telemetry_readers(
    const std::unique_ptr<tt::umd::Cluster>& cluster);

void create_arc_metrics(
    std::vector<std::unique_ptr<BoolMetric>>& bool_metrics,
    std::vector<std::unique_ptr<UIntMetric>>& uint_metrics,
    std::vector<std::unique_ptr<DoubleMetric>>& double_metrics,
    std::vector<std::unique_ptr<StringMetric>>& string_metrics,
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const std::unique_ptr<TopologyHelper>& topology_translation,
    const std::unique_ptr<tt::tt_metal::Hal>& hal,
    const std::unordered_map<tt::ChipId, std::shared_ptr<CachingARCTelemetryReader>>& telemetry_reader_by_chip_id);
