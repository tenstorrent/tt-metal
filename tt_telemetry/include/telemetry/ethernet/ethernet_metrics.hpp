#pragma once

/*
 * telemetry/ethernet/ethernet_metrics.hpp
 *
 * Ethernet channel metrics.
 */

 #include <optional>
 #include <string>
 #include <vector>

 #include <tt-metalium/core_coord.hpp>

 #include <telemetry/metric.hpp>
 #include <telemetry/ethernet/ethernet_endpoint.hpp>

class EthernetEndpointUpMetric: public BoolMetric {
public:
    EthernetEndpointUpMetric(size_t id, const EthernetEndpoint &endpoint)
        : BoolMetric(id)
        , endpoint_(endpoint)
    {
    }

    const std::vector<std::string> telemetry_path() const override;
    void update(const tt::Cluster &cluster) override;

private:
    EthernetEndpoint endpoint_;
};

class EthernetCRCErrorCountMetric: public IntMetric {
public:
    EthernetCRCErrorCountMetric(size_t id, const EthernetEndpoint &endpoint, const tt::Cluster &cluster);

    const std::vector<std::string> telemetry_path() const override;
    void update(const tt::Cluster &cluster) override;

private:
    EthernetEndpoint endpoint_;
    std::optional<tt_cxy_pair> virtual_eth_core_;
    uint32_t crc_addr_;
};