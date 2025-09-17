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
 #include <vector>

#include <third_party/umd/device/api/umd/device/cluster.hpp>
#include <llrt/hal.hpp>

#include <telemetry/metric.hpp>
#include <telemetry/ethernet/ethernet_endpoint.hpp>

class EthernetEndpointUpMetric: public BoolMetric {
public:
    static constexpr std::chrono::seconds FORCE_REFRESH_LINK_STATUS_TIMEOUT{120};

    EthernetEndpointUpMetric(const EthernetEndpoint& endpoint, const std::unique_ptr<tt::tt_metal::Hal>& hal);
    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    EthernetEndpoint endpoint_;
    std::chrono::steady_clock::time_point last_force_refresh_time_;
    uint32_t link_up_addr_;
};

class EthernetCRCErrorCountMetric: public UIntMetric {
public:
    EthernetCRCErrorCountMetric(
        const EthernetEndpoint& endpoint,
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        const std::unique_ptr<tt::tt_metal::Hal>& hal);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    EthernetEndpoint endpoint_;
    std::optional<tt::umd::CoreCoord> ethernet_core_;
    uint32_t crc_addr_;
};

class EthernetRetrainCountMetric: public UIntMetric {
public:
    EthernetRetrainCountMetric(
        const EthernetEndpoint& endpoint,
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        const std::unique_ptr<tt::tt_metal::Hal>& hal);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    EthernetEndpoint endpoint_;
    tt::umd::CoreCoord ethernet_core_;
    uint32_t retrain_count_addr_;
};

class EthernetCorrectedCodewordCountMetric: public UIntMetric {
public:
    EthernetCorrectedCodewordCountMetric(
        const EthernetEndpoint& endpoint,
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        const std::unique_ptr<tt::tt_metal::Hal>& hal);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    EthernetEndpoint endpoint_;
    std::optional<tt::umd::CoreCoord> ethernet_core_;
    uint32_t corr_addr_;
};

class EthernetUncorrectedCodewordCountMetric: public UIntMetric {
public:
    EthernetUncorrectedCodewordCountMetric(
        const EthernetEndpoint& endpoint,
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        const std::unique_ptr<tt::tt_metal::Hal>& hal);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    EthernetEndpoint endpoint_;
    std::optional<tt::umd::CoreCoord> ethernet_core_;
    uint32_t uncorr_addr_;
};

void create_ethernet_metrics(
    std::vector<std::unique_ptr<BoolMetric>>& bool_metrics,
    std::vector<std::unique_ptr<UIntMetric>>& uint_metrics,
    std::vector<std::unique_ptr<DoubleMetric>>& double_metrics,
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const std::unique_ptr<tt::tt_metal::Hal>& hal);
