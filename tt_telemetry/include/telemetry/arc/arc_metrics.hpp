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

#include <llrt/hal.hpp>
#include <umd/device/firmware/firmware_info_provider.hpp>
#include <umd/device/cluster.hpp>

#include <telemetry/metric.hpp>
#include <tt_metal/fabric/physical_system_descriptor.hpp>

class TopologyHelper;

class ARCUintMetric : public UIntMetric {
public:
    // Constructor using FirmwareInfoProvider
    ARCUintMetric(
        tt::tt_metal::ASICDescriptor asic_descriptor,
        tt::umd::FirmwareInfoProvider* firmware_provider,
        const std::string& metric_name,
        std::function<std::optional<uint32_t>()> getter_func,
        MetricUnit units = MetricUnit::UNITLESS);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    tt::tt_metal::ASICDescriptor asic_descriptor_;
    tt::umd::FirmwareInfoProvider* firmware_provider_;
    std::string metric_name_;
    std::function<std::optional<uint32_t>()> getter_func_;
};

class ARCDoubleMetric : public DoubleMetric {
public:
    // Constructor using FirmwareInfoProvider
    ARCDoubleMetric(
        tt::tt_metal::ASICDescriptor asic_descriptor,
        tt::umd::FirmwareInfoProvider* firmware_provider,
        const std::string& metric_name,
        std::function<std::optional<double>()> getter_func,
        MetricUnit units = MetricUnit::UNITLESS);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    tt::tt_metal::ASICDescriptor asic_descriptor_;
    tt::umd::FirmwareInfoProvider* firmware_provider_;
    std::string metric_name_;
    std::function<std::optional<double>()> getter_func_;
};

void create_arc_metrics(
    std::vector<std::unique_ptr<BoolMetric>>& bool_metrics,
    std::vector<std::unique_ptr<UIntMetric>>& uint_metrics,
    std::vector<std::unique_ptr<DoubleMetric>>& double_metrics,
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const std::unique_ptr<TopologyHelper>& topology_translation,
    const std::unique_ptr<tt::tt_metal::Hal>& hal);
