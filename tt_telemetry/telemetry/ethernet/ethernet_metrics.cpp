// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <telemetry/ethernet/ethernet_metrics.hpp>
#include <telemetry/ethernet/ethernet_helpers.hpp>
#include <chrono>


/**************************************************************************************************
 EthernetEndpointUpMetric

 Whether link is up.
**************************************************************************************************/

EthernetEndpointUpMetric::EthernetEndpointUpMetric(size_t id, const EthernetEndpoint &endpoint, const std::unique_ptr<tt::tt_metal::Hal> &hal)
    : BoolMetric(id)
    , endpoint_(endpoint)
    , last_force_refresh_time_(std::chrono::steady_clock::time_point::min())
    , link_up_addr_(hal->get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::LINK_UP))
{
}

const std::vector<std::string> EthernetEndpointUpMetric::telemetry_path() const {
    std::vector<std::string> path = endpoint_.telemetry_path();
    path.push_back("linkIsUp");
    return path;
}

void EthernetEndpointUpMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    // Check if enough time has elapsed since last force refresh
    bool should_force_refresh = (start_of_update_cycle - last_force_refresh_time_) >= FORCE_REFRESH_LINK_STATUS_TIMEOUT;

    // If we're forcing a refresh, update the timestamp
    if (should_force_refresh) {
        last_force_refresh_time_ = start_of_update_cycle;
    }

    bool is_up_now = is_ethernet_endpoint_up(cluster, endpoint_, link_up_addr_, should_force_refresh);
    bool is_up_old = value_;
    changed_since_transmission_ = is_up_now != is_up_old;
    value_ = is_up_now;
    timestamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

/**************************************************************************************************
 EthernetCRCErrorCountMetric

 Number of CRC errors encountered.
**************************************************************************************************/

EthernetCRCErrorCountMetric::EthernetCRCErrorCountMetric(
    size_t id, const EthernetEndpoint& endpoint, const std::unique_ptr<tt::umd::Cluster>& cluster, const std::unique_ptr<tt::tt_metal::Hal> &hal) :
    UIntMetric(id), endpoint_(endpoint) {
    value_ = 0;
    tt::umd::TTDevice* device = cluster->get_tt_device(endpoint_.chip.id);
    TT_FATAL(device->get_arch() == tt::ARCH::WORMHOLE_B0, "Metric {} available only on Wormhole", __func__);
    ethernet_core_ = tt::umd::CoreCoord(
        endpoint_.ethernet_core.x,
        endpoint_.ethernet_core.y,
        tt::umd::CoreType::ETH,
        tt::umd::CoordSystem::LOGICAL);
    crc_addr_ = hal->get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::CRC_ERR);
}

const std::vector<std::string> EthernetCRCErrorCountMetric::telemetry_path() const {
    std::vector<std::string> path = endpoint_.telemetry_path();
    path.push_back("crcErrorCount");
    return path;
}

void EthernetCRCErrorCountMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    if (!ethernet_core_.has_value()) {
        // Architecture not yet supported
        return;
    }
    uint32_t crc_error_val = 0;
    cluster->read_from_device(&crc_error_val, endpoint_.chip.id, ethernet_core_.value(), crc_addr_, sizeof(uint32_t));
    value_ = uint64_t(crc_error_val);
    timestamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

/**************************************************************************************************
 EthernetRetrainCountMetric

 Number of CRC retrains that occurred.
**************************************************************************************************/

EthernetRetrainCountMetric::EthernetRetrainCountMetric(
    size_t id, const EthernetEndpoint& endpoint, const std::unique_ptr<tt::umd::Cluster>& cluster, const std::unique_ptr<tt::tt_metal::Hal> &hal) :
    UIntMetric(id), endpoint_(endpoint) {
    value_ = 0;
    ethernet_core_ = tt::umd::CoreCoord(
        endpoint_.ethernet_core.x, endpoint_.ethernet_core.y, tt::umd::CoreType::ETH, tt::umd::CoordSystem::LOGICAL);
    retrain_count_addr_ = hal->get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::RETRAIN_COUNT);
}

const std::vector<std::string> EthernetRetrainCountMetric::telemetry_path() const {
    std::vector<std::string> path = endpoint_.telemetry_path();
    path.push_back("retrainCount");
    return path;
}

void EthernetRetrainCountMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    uint32_t data = 0;
    cluster->read_from_device(&data, endpoint_.chip.id, ethernet_core_, retrain_count_addr_, sizeof(uint32_t));
    value_ = uint64_t(data);
    timestamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

/**************************************************************************************************
 EthernetCorrectedCodewordCountMetric

 Number of codeword corrections.
**************************************************************************************************/

EthernetCorrectedCodewordCountMetric::EthernetCorrectedCodewordCountMetric(
    size_t id, const EthernetEndpoint& endpoint, const std::unique_ptr<tt::umd::Cluster>& cluster, const std::unique_ptr<tt::tt_metal::Hal> &hal) :
    UIntMetric(id), endpoint_(endpoint) {
    value_ = 0;
    tt::umd::TTDevice* device = cluster->get_tt_device(endpoint_.chip.id);
    TT_FATAL(device->get_arch() == tt::ARCH::WORMHOLE_B0, "Metric {} available only on Wormhole", __func__);
    ethernet_core_ = tt::umd::CoreCoord(
        endpoint_.ethernet_core.x,
        endpoint_.ethernet_core.y,
        tt::umd::CoreType::ETH,
        tt::umd::CoordSystem::LOGICAL);
    corr_addr_ = hal->get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::CORR_CW);
}

const std::vector<std::string> EthernetCorrectedCodewordCountMetric::telemetry_path() const {
    std::vector<std::string> path = endpoint_.telemetry_path();
    path.push_back("correctedCodewordCount");
    return path;
}

void EthernetCorrectedCodewordCountMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    if (!ethernet_core_.has_value()) {
        // Architecture not yet supported
        return;
    }
    uint32_t hi = 0;
    uint32_t lo = 0;
    cluster->read_from_device(&hi, endpoint_.chip.id, ethernet_core_.value(), corr_addr_ + 0, sizeof(uint32_t));
    cluster->read_from_device(&lo, endpoint_.chip.id, ethernet_core_.value(), corr_addr_ + 4, sizeof(uint32_t));
    value_ = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
    timestamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

/**************************************************************************************************
 EthernetUncorrectedCodewordCountMetric

 Number of uncorrected codewords.
**************************************************************************************************/

EthernetUncorrectedCodewordCountMetric::EthernetUncorrectedCodewordCountMetric(
    size_t id, const EthernetEndpoint& endpoint, const std::unique_ptr<tt::umd::Cluster>& cluster, const std::unique_ptr<tt::tt_metal::Hal> &hal) :
    UIntMetric(id), endpoint_(endpoint) {
    value_ = 0;
    tt::umd::TTDevice* device = cluster->get_tt_device(endpoint_.chip.id);
    TT_FATAL(device->get_arch() == tt::ARCH::WORMHOLE_B0, "Metric {} available only on Wormhole", __func__);
    ethernet_core_ = tt::umd::CoreCoord(
        endpoint_.ethernet_core.x,
        endpoint_.ethernet_core.y,
        tt::umd::CoreType::ETH,
        tt::umd::CoordSystem::LOGICAL);
    uncorr_addr_ = hal->get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNCORR_CW);
}

const std::vector<std::string> EthernetUncorrectedCodewordCountMetric::telemetry_path() const {
    std::vector<std::string> path = endpoint_.telemetry_path();
    path.push_back("uncorrectedCodewordCount");
    return path;
}

void EthernetUncorrectedCodewordCountMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    if (!ethernet_core_.has_value()) {
        // Architecture not yet supported
        return;
    }
    uint32_t hi = 0;
    uint32_t lo = 0;
    cluster->read_from_device(&hi, endpoint_.chip.id, ethernet_core_.value(), uncorr_addr_ + 0, sizeof(uint32_t));
    cluster->read_from_device(&lo, endpoint_.chip.id, ethernet_core_.value(), uncorr_addr_ + 4, sizeof(uint32_t));
    value_ = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
    timestamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}
