#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/ethernet/ethernet_metrics.hpp
 *
 * Ethernet endpoint (i.e., a core or channel) metrics.
 */

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <third_party/umd/device/api/umd/device/cluster.hpp>
#include <llrt/hal.hpp>
#include <tt_metal/fabric/physical_system_descriptor.hpp>

#include <telemetry/metric.hpp>
#include <telemetry/ethernet/fabric_telemetry_reader.hpp>

namespace tt::scaleout_tools::fsd::proto {
class FactorySystemDescriptor;
}

class TopologyHelper;

class EthernetEndpointUpMetric: public BoolMetric {
public:
    static constexpr std::chrono::seconds FORCE_REFRESH_LINK_STATUS_TIMEOUT{120};

    EthernetEndpointUpMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        tt::ChipId chip_id,
        uint32_t channel,
        const std::unique_ptr<tt::tt_metal::Hal>& hal);
    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    tt::ChipId chip_id_;
    uint32_t channel_;
    std::chrono::steady_clock::time_point last_force_refresh_time_;
    uint32_t link_up_addr_;
};

class EthernetCRCErrorCountMetric: public UIntMetric {
public:
    EthernetCRCErrorCountMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        tt::ChipId chip_id,
        uint32_t channel,
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        const std::unique_ptr<tt::tt_metal::Hal>& hal);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    tt::ChipId chip_id_;
    uint32_t channel_;
    tt::umd::CoreCoord ethernet_core_;
    uint32_t crc_addr_;
};

class EthernetRetrainCountMetric: public UIntMetric {
public:
    EthernetRetrainCountMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        tt::ChipId chip_id,
        uint32_t channel,
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        const std::unique_ptr<tt::tt_metal::Hal>& hal);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    tt::ChipId chip_id_;
    uint32_t channel_;
    tt::umd::CoreCoord ethernet_core_;
    uint32_t retrain_count_addr_;
};

class EthernetCorrectedCodewordCountMetric: public UIntMetric {
public:
    EthernetCorrectedCodewordCountMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        tt::ChipId chip_id,
        uint32_t channel,
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        const std::unique_ptr<tt::tt_metal::Hal>& hal);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    tt::ChipId chip_id_;
    uint32_t channel_;
    tt::umd::CoreCoord ethernet_core_;
    uint32_t corr_addr_;
};

class EthernetUncorrectedCodewordCountMetric: public UIntMetric {
public:
    EthernetUncorrectedCodewordCountMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        tt::ChipId chip_id,
        uint32_t channel,
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        const std::unique_ptr<tt::tt_metal::Hal>& hal);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    tt::ChipId chip_id_;
    uint32_t channel_;
    tt::umd::CoreCoord ethernet_core_;
    uint32_t uncorr_addr_;
};

class FabricBandwidthMetric: public DoubleMetric {
public:
    FabricBandwidthMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        tt::ChipId chip_id,
        uint32_t channel,
        size_t erisc_core,
        std::shared_ptr<FabricTelemetryReader> telemetry_reader);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    tt::ChipId chip_id_;
    uint32_t channel_;
    size_t erisc_core_;
    std::shared_ptr<FabricTelemetryReader> telemetry_reader_;
};

class FabricTxWordsMetric: public UIntMetric {
public:
    FabricTxWordsMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        std::shared_ptr<FabricTelemetryReader> telemetry_reader);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    std::shared_ptr<FabricTelemetryReader> telemetry_reader_;
};

class FabricTxPacketsMetric: public UIntMetric {
public:
    FabricTxPacketsMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        std::shared_ptr<FabricTelemetryReader> telemetry_reader);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    std::shared_ptr<FabricTelemetryReader> telemetry_reader_;
};

class FabricHeartbeatTxMetric: public UIntMetric {
public:
    FabricHeartbeatTxMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        size_t erisc_core,
        std::shared_ptr<FabricTelemetryReader> telemetry_reader);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    size_t erisc_core_;
    std::shared_ptr<FabricTelemetryReader> telemetry_reader_;
};

class FabricRxWordsMetric: public UIntMetric {
public:
    FabricRxWordsMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        std::shared_ptr<FabricTelemetryReader> telemetry_reader);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    std::shared_ptr<FabricTelemetryReader> telemetry_reader_;
};

class FabricRxPacketsMetric: public UIntMetric {
public:
    FabricRxPacketsMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        std::shared_ptr<FabricTelemetryReader> telemetry_reader);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    std::shared_ptr<FabricTelemetryReader> telemetry_reader_;
};

class FabricHeartbeatRxMetric: public UIntMetric {
public:
    FabricHeartbeatRxMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        size_t erisc_core,
        std::shared_ptr<FabricTelemetryReader> telemetry_reader);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    size_t erisc_core_;
    std::shared_ptr<FabricTelemetryReader> telemetry_reader_;
};

class FabricRouterStateMetric: public UIntMetric {
public:
    FabricRouterStateMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        size_t erisc_core,
        std::shared_ptr<FabricTelemetryReader> telemetry_reader);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    size_t erisc_core_;
    std::shared_ptr<FabricTelemetryReader> telemetry_reader_;
};

class FabricMeshIDMetric: public UIntMetric {
public:
    FabricMeshIDMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        std::shared_ptr<FabricTelemetryReader> telemetry_reader);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    std::shared_ptr<FabricTelemetryReader> telemetry_reader_;
};

class FabricDeviceIDMetric: public UIntMetric {
public:
    FabricDeviceIDMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        std::shared_ptr<FabricTelemetryReader> telemetry_reader);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    std::shared_ptr<FabricTelemetryReader> telemetry_reader_;
};

class FabricDirectionMetric: public StringMetric {
public:
    FabricDirectionMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        std::shared_ptr<FabricTelemetryReader> telemetry_reader);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    std::shared_ptr<FabricTelemetryReader> telemetry_reader_;
    uint8_t raw_value_;
};

class FabricConfigMetric: public StringMetric {
public:
    FabricConfigMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        std::shared_ptr<FabricTelemetryReader> telemetry_reader);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    std::shared_ptr<FabricTelemetryReader> telemetry_reader_;
    uint32_t raw_value_;
};

void create_ethernet_metrics(
    std::vector<std::unique_ptr<BoolMetric>>& bool_metrics,
    std::vector<std::unique_ptr<UIntMetric>>& uint_metrics,
    std::vector<std::unique_ptr<DoubleMetric>>& double_metrics,
    std::vector<std::unique_ptr<StringMetric>>& string_metrics,
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const tt::scaleout_tools::fsd::proto::FactorySystemDescriptor& fsd,
    const std::unique_ptr<TopologyHelper>& topology_translation,
    const std::unique_ptr<tt::tt_metal::Hal>& hal);
