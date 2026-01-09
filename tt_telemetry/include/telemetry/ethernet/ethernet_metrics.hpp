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
 #include <optional>
 #include <string>
#include <unordered_map>
#include <vector>

#include <third_party/umd/device/api/umd/device/cluster.hpp>
#include <llrt/hal.hpp>
#include <tt_metal/fabric/physical_system_descriptor.hpp>

#include <telemetry/metric.hpp>
#include <topology/topology.hpp>

namespace tt::scaleout_tools::fsd::proto {
class FactorySystemDescriptor;
}

class CachingFabricTelemetryReader;
class CachingARCTelemetryReader;

class EthernetEndpointUpMetric : public BoolMetric {
public:
    static constexpr std::chrono::seconds FORCE_REFRESH_LINK_STATUS_TIMEOUT{120};

    EthernetEndpointUpMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        tt::ChipId chip_id,
        uint32_t channel,
        const std::unique_ptr<tt::tt_metal::Hal>& hal,
        const std::unique_ptr<TopologyHelper>& topology_helper = nullptr);
    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;
    std::unordered_map<std::string, std::string> labels() const override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    tt::ChipId chip_id_;
    uint32_t channel_;
    std::chrono::steady_clock::time_point last_force_refresh_time_;
    uint32_t link_up_addr_;
    std::optional<PhysicalLinkInfo> link_info_;
};

class EthernetCRCErrorCountMetric : public UIntMetric {
public:
    EthernetCRCErrorCountMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        tt::ChipId chip_id,
        uint32_t channel,
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        const std::unique_ptr<tt::tt_metal::Hal>& hal,
        const std::unique_ptr<TopologyHelper>& topology_helper = nullptr);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;
    std::unordered_map<std::string, std::string> labels() const override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    tt::ChipId chip_id_;
    uint32_t channel_;
    tt::umd::CoreCoord ethernet_core_;
    uint32_t crc_addr_;
    std::optional<PhysicalLinkInfo> link_info_;
};

class EthernetRetrainCountMetric : public UIntMetric {
public:
    EthernetRetrainCountMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        tt::ChipId chip_id,
        uint32_t channel,
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        const std::unique_ptr<tt::tt_metal::Hal>& hal,
        const std::unique_ptr<TopologyHelper>& topology_helper = nullptr);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;
    std::unordered_map<std::string, std::string> labels() const override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    tt::ChipId chip_id_;
    uint32_t channel_;
    tt::umd::CoreCoord ethernet_core_;
    uint32_t retrain_count_addr_;
    std::optional<PhysicalLinkInfo> link_info_;
};

class EthernetCorrectedCodewordCountMetric : public UIntMetric {
public:
    EthernetCorrectedCodewordCountMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        tt::ChipId chip_id,
        uint32_t channel,
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        const std::unique_ptr<tt::tt_metal::Hal>& hal,
        const std::unique_ptr<TopologyHelper>& topology_helper = nullptr);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;
    std::unordered_map<std::string, std::string> labels() const override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    tt::ChipId chip_id_;
    uint32_t channel_;
    tt::umd::CoreCoord ethernet_core_;
    uint32_t corr_addr_;
    std::optional<PhysicalLinkInfo> link_info_;
};

class EthernetUncorrectedCodewordCountMetric : public UIntMetric {
public:
    EthernetUncorrectedCodewordCountMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        tt::ChipId chip_id,
        uint32_t channel,
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        const std::unique_ptr<tt::tt_metal::Hal>& hal,
        const std::unique_ptr<TopologyHelper>& topology_helper = nullptr);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;
    std::unordered_map<std::string, std::string> labels() const override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    tt::ChipId chip_id_;
    uint32_t channel_;
    tt::umd::CoreCoord ethernet_core_;
    uint32_t uncorr_addr_;
    std::optional<PhysicalLinkInfo> link_info_;
};

class FabricMeshIdMetric : public UIntMetric {
public:
    FabricMeshIdMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader,
        const std::unique_ptr<TopologyHelper>& topology_helper = nullptr);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;
    std::unordered_map<std::string, std::string> labels() const override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader_;
    std::optional<PhysicalLinkInfo> link_info_;
};

class FabricDeviceIdMetric : public UIntMetric {
public:
    FabricDeviceIdMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader,
        const std::unique_ptr<TopologyHelper>& topology_helper = nullptr);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;
    std::unordered_map<std::string, std::string> labels() const override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader_;
    std::optional<PhysicalLinkInfo> link_info_;
};

class FabricDirectionMetric : public UIntMetric {
public:
    FabricDirectionMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader,
        const std::unique_ptr<TopologyHelper>& topology_helper = nullptr);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;
    std::unordered_map<std::string, std::string> labels() const override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader_;
    std::optional<PhysicalLinkInfo> link_info_;
};

class FabricConfigMetric : public UIntMetric {
public:
    FabricConfigMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader,
        const std::unique_ptr<TopologyHelper>& topology_helper = nullptr);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;
    std::unordered_map<std::string, std::string> labels() const override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader_;
    std::optional<PhysicalLinkInfo> link_info_;
};

// Fabric telemetry metrics - raw counters from device
class FabricTxWordsMetric : public UIntMetric {
public:
    FabricTxWordsMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader,
        const std::unique_ptr<TopologyHelper>& topology_helper = nullptr);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;
    std::unordered_map<std::string, std::string> labels() const override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader_;
    std::optional<PhysicalLinkInfo> link_info_;
};

class FabricRxWordsMetric : public UIntMetric {
public:
    FabricRxWordsMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader,
        const std::unique_ptr<TopologyHelper>& topology_helper = nullptr);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;
    std::unordered_map<std::string, std::string> labels() const override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader_;
    std::optional<PhysicalLinkInfo> link_info_;
};

class FabricTxPacketsMetric : public UIntMetric {
public:
    FabricTxPacketsMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader,
        const std::unique_ptr<TopologyHelper>& topology_helper = nullptr);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;
    std::unordered_map<std::string, std::string> labels() const override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader_;
    std::optional<PhysicalLinkInfo> link_info_;
};

class FabricRxPacketsMetric : public UIntMetric {
public:
    FabricRxPacketsMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader,
        const std::unique_ptr<TopologyHelper>& topology_helper = nullptr);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;
    std::unordered_map<std::string, std::string> labels() const override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader_;
    std::optional<PhysicalLinkInfo> link_info_;
};

class FabricSupportedStatsMetric : public UIntMetric {
public:
    FabricSupportedStatsMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader,
        const std::unique_ptr<TopologyHelper>& topology_helper = nullptr);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;
    std::unordered_map<std::string, std::string> labels() const override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader_;
    std::optional<PhysicalLinkInfo> link_info_;
};

class FabricTxBandwidthMetric : public DoubleMetric {
public:
    FabricTxBandwidthMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader,
        const std::unique_ptr<TopologyHelper>& topology_helper = nullptr,
        std::shared_ptr<CachingARCTelemetryReader> arc_telemetry_reader = nullptr,
        tt::ARCH arch = tt::ARCH::Invalid);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;
    std::unordered_map<std::string, std::string> labels() const override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader_;
    std::optional<PhysicalLinkInfo> link_info_;
    std::shared_ptr<CachingARCTelemetryReader> arc_telemetry_reader_;
    tt::ARCH arch_;

    uint64_t prev_words_ = 0;
    uint64_t prev_cycles_ = 0;
    bool first_update_ = true;
};

class FabricRxBandwidthMetric : public DoubleMetric {
public:
    FabricRxBandwidthMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader,
        const std::unique_ptr<TopologyHelper>& topology_helper = nullptr,
        std::shared_ptr<CachingARCTelemetryReader> arc_telemetry_reader = nullptr,
        tt::ARCH arch = tt::ARCH::Invalid);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;
    std::unordered_map<std::string, std::string> labels() const override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader_;
    std::optional<PhysicalLinkInfo> link_info_;
    std::shared_ptr<CachingARCTelemetryReader> arc_telemetry_reader_;
    tt::ARCH arch_;

    uint64_t prev_words_ = 0;
    uint64_t prev_cycles_ = 0;
    bool first_update_ = true;
};

class FabricTxPeakBandwidthMetric : public DoubleMetric {
public:
    FabricTxPeakBandwidthMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader,
        const std::unique_ptr<TopologyHelper>& topology_helper = nullptr,
        std::shared_ptr<CachingARCTelemetryReader> arc_telemetry_reader = nullptr,
        tt::ARCH arch = tt::ARCH::Invalid);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;
    std::unordered_map<std::string, std::string> labels() const override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader_;
    std::optional<PhysicalLinkInfo> link_info_;
    std::shared_ptr<CachingARCTelemetryReader> arc_telemetry_reader_;
    tt::ARCH arch_;

    uint64_t prev_words_ = 0;
    uint64_t prev_cycles_ = 0;
    bool first_update_ = true;
};

class FabricRxPeakBandwidthMetric : public DoubleMetric {
public:
    FabricRxPeakBandwidthMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader,
        const std::unique_ptr<TopologyHelper>& topology_helper = nullptr,
        std::shared_ptr<CachingARCTelemetryReader> arc_telemetry_reader = nullptr,
        tt::ARCH arch = tt::ARCH::Invalid);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;
    std::unordered_map<std::string, std::string> labels() const override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader_;
    std::optional<PhysicalLinkInfo> link_info_;
    std::shared_ptr<CachingARCTelemetryReader> arc_telemetry_reader_;
    tt::ARCH arch_;

    uint64_t prev_words_ = 0;
    uint64_t prev_cycles_ = 0;
    bool first_update_ = true;
};

class FabricTxHeartbeatMetric : public UIntMetric {
public:
    FabricTxHeartbeatMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        size_t erisc_core,
        std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader,
        const std::unique_ptr<TopologyHelper>& topology_helper = nullptr);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;
    std::unordered_map<std::string, std::string> labels() const override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    size_t erisc_core_;
    std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader_;
    std::optional<PhysicalLinkInfo> link_info_;
};

class FabricRxHeartbeatMetric : public UIntMetric {
public:
    FabricRxHeartbeatMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        size_t erisc_core,
        std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader,
        const std::unique_ptr<TopologyHelper>& topology_helper = nullptr);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;
    std::unordered_map<std::string, std::string> labels() const override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    size_t erisc_core_;
    std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader_;
    std::optional<PhysicalLinkInfo> link_info_;
};

class FabricRouterStateMetric : public UIntMetric {
public:
    FabricRouterStateMetric(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        uint32_t channel,
        size_t erisc_core,
        std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader,
        const std::unique_ptr<TopologyHelper>& topology_helper = nullptr);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;
    std::unordered_map<std::string, std::string> labels() const override;

private:
    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    uint32_t channel_;
    size_t erisc_core_;
    std::shared_ptr<CachingFabricTelemetryReader> telemetry_reader_;
    std::optional<PhysicalLinkInfo> link_info_;
};

void create_ethernet_metrics(
    std::vector<std::unique_ptr<BoolMetric>>& bool_metrics,
    std::vector<std::unique_ptr<UIntMetric>>& uint_metrics,
    std::vector<std::unique_ptr<DoubleMetric>>& double_metrics,
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const tt::scaleout_tools::fsd::proto::FactorySystemDescriptor& fsd,
    const std::unique_ptr<TopologyHelper>& topology_translation,
    const std::unique_ptr<tt::tt_metal::Hal>& hal,
    const std::unordered_map<tt::ChipId, std::shared_ptr<CachingARCTelemetryReader>>& arc_telemetry_reader_by_chip_id,
    bool mmio_only = false);
