#include "impl/context/metal_context.hpp"

#include <telemetry/ethernet/ethernet_metrics.hpp>
#include <telemetry/ethernet/ethernet_helpers.hpp>


/**************************************************************************************************
 EthernetEndpointUpMetric

 Whether link is up.
**************************************************************************************************/

const std::vector<std::string> EthernetEndpointUpMetric::telemetry_path() const {
    std::vector<std::string> path = endpoint_.telemetry_path();
    path.push_back("linkIsUp");
    return path;
}

void EthernetEndpointUpMetric::update(const tt::Cluster &cluster) {
    bool is_up_now = is_ethernet_endpoint_up(cluster, endpoint_);
    bool is_up_old = value_;
    changed_since_transmission_ = is_up_now != is_up_old;
    value_ = is_up_now;
}


/**************************************************************************************************
 EthernetCRCErrorCountMetric

 Number of CRC errors encountered.
**************************************************************************************************/

EthernetCRCErrorCountMetric::EthernetCRCErrorCountMetric(
    size_t id, const EthernetEndpoint& endpoint, const tt::Cluster& cluster) :
    UIntMetric(id), endpoint_(endpoint) {
    value_ = 0;
    if (cluster.arch() == tt::ARCH::WORMHOLE_B0) {
        virtual_eth_core_ = tt_cxy_pair(endpoint.chip.id, cluster.get_virtual_coordinate_from_logical_coordinates(endpoint.chip.id, endpoint.ethernet_core, CoreType::ETH));
        crc_addr_ = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::CRC_ERR);
    }
}

const std::vector<std::string> EthernetCRCErrorCountMetric::telemetry_path() const {
    std::vector<std::string> path = endpoint_.telemetry_path();
    path.push_back("crcErrorCount");
    return path;
}

void EthernetCRCErrorCountMetric::update(const tt::Cluster &cluster) {
    if (!virtual_eth_core_.has_value()) {
        // Architecture not yet supported
        return;
    }
    uint32_t crc_error_val = 0;
    cluster.read_core(&crc_error_val, sizeof(uint32_t), virtual_eth_core_.value(), crc_addr_);
    value_ = uint64_t(crc_error_val);
}


/**************************************************************************************************
 EthernetRetrainCountMetric

 Number of CRC retrains that occurred.
**************************************************************************************************/

EthernetRetrainCountMetric::EthernetRetrainCountMetric(
    size_t id, const EthernetEndpoint& endpoint, const tt::Cluster& cluster) :
    UIntMetric(id), endpoint_(endpoint) {
    value_ = 0;
    virtual_eth_core_ = tt_cxy_pair(endpoint.chip.id, cluster.get_virtual_coordinate_from_logical_coordinates(endpoint.chip.id, endpoint.ethernet_core, CoreType::ETH));
    retrain_count_addr_ = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::RETRAIN_COUNT);
}

const std::vector<std::string> EthernetRetrainCountMetric::telemetry_path() const {
    std::vector<std::string> path = endpoint_.telemetry_path();
    path.push_back("retrainCount");
    return path;
}

void EthernetRetrainCountMetric::update(const tt::Cluster &cluster) {
    std::vector<uint32_t> data;
    cluster.read_core(data, sizeof(uint32_t), virtual_eth_core_, retrain_count_addr_);
    if (data.size() >= 1) {
        value_ = uint64_t(data[0]);
    }
}


/**************************************************************************************************
 EthernetCorrectedCodewordCountMetric

 Number of codeword corrections.
**************************************************************************************************/

EthernetCorrectedCodewordCountMetric::EthernetCorrectedCodewordCountMetric(
    size_t id, const EthernetEndpoint& endpoint, const tt::Cluster& cluster) :
    UIntMetric(id), endpoint_(endpoint) {
    value_ = 0;
    if (cluster.arch() == tt::ARCH::WORMHOLE_B0) {
        virtual_eth_core_ = tt_cxy_pair(endpoint.chip.id, cluster.get_virtual_coordinate_from_logical_coordinates(endpoint.chip.id, endpoint.ethernet_core, CoreType::ETH));
        corr_addr_ = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::CORR_CW);
    }
}

const std::vector<std::string> EthernetCorrectedCodewordCountMetric::telemetry_path() const {
    std::vector<std::string> path = endpoint_.telemetry_path();
    path.push_back("correctedCodewordCount");
    return path;
}

void EthernetCorrectedCodewordCountMetric::update(const tt::Cluster &cluster) {
    if (!virtual_eth_core_.has_value()) {
        // Architecture not yet supported
        return;
    }
    uint32_t hi = 0;
    uint32_t lo = 0;
    cluster.read_core(&hi, sizeof(uint32_t), virtual_eth_core_.value(), corr_addr_ + 0);
    cluster.read_core(&lo, sizeof(uint32_t), virtual_eth_core_.value(), corr_addr_ + 4);
    value_ = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}


/**************************************************************************************************
 EthernetUncorrectedCodewordCountMetric

 Number of uncorrected codewords.
**************************************************************************************************/

EthernetUncorrectedCodewordCountMetric::EthernetUncorrectedCodewordCountMetric(
    size_t id, const EthernetEndpoint& endpoint, const tt::Cluster& cluster) :
    UIntMetric(id), endpoint_(endpoint) {
    value_ = 0;
    if (cluster.arch() == tt::ARCH::WORMHOLE_B0) {
        virtual_eth_core_ = tt_cxy_pair(endpoint.chip.id, cluster.get_virtual_coordinate_from_logical_coordinates(endpoint.chip.id, endpoint.ethernet_core, CoreType::ETH));
        uncorr_addr_ = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNCORR_CW);
    }
}

const std::vector<std::string> EthernetUncorrectedCodewordCountMetric::telemetry_path() const {
    std::vector<std::string> path = endpoint_.telemetry_path();
    path.push_back("uncorrectedCodewordCount");
    return path;
}

void EthernetUncorrectedCodewordCountMetric::update(const tt::Cluster &cluster) {
    if (!virtual_eth_core_.has_value()) {
        // Architecture not yet supported
        return;
    }
    uint32_t hi = 0;
    uint32_t lo = 0;
    cluster.read_core(&hi, sizeof(uint32_t), virtual_eth_core_.value(), uncorr_addr_ + 0);
    cluster.read_core(&lo, sizeof(uint32_t), virtual_eth_core_.value(), uncorr_addr_ + 4);
    value_ = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}
