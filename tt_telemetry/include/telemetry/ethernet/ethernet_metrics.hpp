#pragma once

/*
 * telemetry/ethernet/ethernet_metrics.hpp
 *
 * Ethernet endpoint (i.e., a core or channel) metrics.
 */

 #include <chrono>
 #include <optional>
 #include <string>
 #include <vector>

 #include <tt-metalium/core_coord.hpp>

 #include <telemetry/metric.hpp>
 #include <telemetry/ethernet/ethernet_endpoint.hpp>

class EthernetEndpointUpMetric: public BoolMetric {
public:
    static constexpr std::chrono::seconds FORCE_REFRESH_LINK_STATUS_TIMEOUT{120};

    EthernetEndpointUpMetric(size_t id, const EthernetEndpoint &endpoint)
        : BoolMetric(id)
        , endpoint_(endpoint)
        , last_force_refresh_time_(std::chrono::steady_clock::time_point::min())
    {
    }

    const std::vector<std::string> telemetry_path() const override;
    void update(const tt::Cluster &cluster, std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    EthernetEndpoint endpoint_;
    std::chrono::steady_clock::time_point last_force_refresh_time_;
};

class EthernetCRCErrorCountMetric: public UIntMetric {
public:
    EthernetCRCErrorCountMetric(size_t id, const EthernetEndpoint &endpoint, const tt::Cluster &cluster);

    const std::vector<std::string> telemetry_path() const override;
    void update(const tt::Cluster &cluster, std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    EthernetEndpoint endpoint_;
    std::optional<tt_cxy_pair> virtual_eth_core_;
    uint32_t crc_addr_;
};

class EthernetRetrainCountMetric: public UIntMetric {
public:
    EthernetRetrainCountMetric(size_t id, const EthernetEndpoint &endpoint, const tt::Cluster &cluster);

    const std::vector<std::string> telemetry_path() const override;
    void update(const tt::Cluster &cluster, std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    EthernetEndpoint endpoint_;
    tt_cxy_pair virtual_eth_core_;
    uint32_t retrain_count_addr_;
};

class EthernetCorrectedCodewordCountMetric: public UIntMetric {
public:
    EthernetCorrectedCodewordCountMetric(size_t id, const EthernetEndpoint &endpoint, const tt::Cluster &cluster);

    const std::vector<std::string> telemetry_path() const override;
    void update(const tt::Cluster &cluster, std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    EthernetEndpoint endpoint_;
    std::optional<tt_cxy_pair> virtual_eth_core_;
    uint32_t corr_addr_;
};

class EthernetUncorrectedCodewordCountMetric: public UIntMetric {
public:
    EthernetUncorrectedCodewordCountMetric(size_t id, const EthernetEndpoint &endpoint, const tt::Cluster &cluster);

    const std::vector<std::string> telemetry_path() const override;
    void update(const tt::Cluster &cluster, std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    EthernetEndpoint endpoint_;
    std::optional<tt_cxy_pair> virtual_eth_core_;
    uint32_t uncorr_addr_;
};
