#include <unistd.h>

#include <telemetry/ethernet/ethernet_metrics.hpp>
#include <telemetry/ethernet/ethernet_helpers.hpp>

const std::vector<std::string> EthernetEndpointUpMetric::telemetry_path() const {
    return endpoint_.telemetry_path();
}

void EthernetEndpointUpMetric::update(const tt::Cluster &cluster) {
    bool is_up_now = is_ethernet_endpoint_up(cluster, endpoint_);
    bool is_up_old = value_;
    changed_since_transmission_ = is_up_now != is_up_old;
    value_ = is_up_now;
}