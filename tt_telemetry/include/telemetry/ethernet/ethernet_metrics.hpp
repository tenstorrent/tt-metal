#pragma once

/*
 * telemetry/ethernet/ethernet_metrics.hpp
 *
 * Ethernet channel metrics.
 */

 #include <string>
 #include <vector>

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