// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cstring>
#include <bit>
#include <array>

#include "protobuf/factory_system_descriptor.pb.h"

#include <telemetry/ethernet/ethernet_metrics.hpp>
#include <telemetry/ethernet/ethernet_helpers.hpp>
#include <telemetry/ethernet/fabric_bandwidth_telemetry_reader.hpp>
#include <topology/topology.hpp>

/**************************************************************************************************
 Metric Creation
**************************************************************************************************/

// Creates Ethernet metrics with contiguous IDs and returns the next free ID value
void create_ethernet_metrics(
    std::vector<std::unique_ptr<BoolMetric>>& bool_metrics,
    std::vector<std::unique_ptr<UIntMetric>>& uint_metrics,
    std::vector<std::unique_ptr<DoubleMetric>>& double_metrics,
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const tt::scaleout_tools::fsd::proto::FactorySystemDescriptor& fsd,
    const std::unique_ptr<TopologyHelper>& topology_translation,
    const std::unique_ptr<tt::tt_metal::Hal>& hal) {
    log_info(tt::LogAlways, "Creating Ethernet metrics...");

    // Get all the Ethernet endpoints on this host that should be present in this cluster according
    // to its factory system descriptor
    std::vector<tt::scaleout_tools::fsd::proto::FactorySystemDescriptor_EndPoint> endpoints;
    for (const auto& connection : fsd.eth_connections().connection()) {
        std::string hostname_a = fsd.hosts()[connection.endpoint_a().host_id()].hostname();
        std::string hostname_b = fsd.hosts()[connection.endpoint_b().host_id()].hostname();
        log_info(
            tt::LogAlways,
            "Found endpoint in FSD: A: hostname={}, tray_id={}, asic_location={}, channel={}",
            hostname_a,
            connection.endpoint_a().tray_id(),
            connection.endpoint_a().asic_location(),
            connection.endpoint_a().chan_id());
        log_info(
            tt::LogAlways,
            "Found endpoint in FSD: B: hostname={}, tray_id={}, asic_location={}, channel={}",
            hostname_b,
            connection.endpoint_b().tray_id(),
            connection.endpoint_b().asic_location(),
            connection.endpoint_b().chan_id());

        // a and b are each unique (that is, nothing listed as b will ever appear as a)
        if (hostname_a == topology_translation->my_host_name) {
            endpoints.push_back(connection.endpoint_a());
        }
        if (hostname_b == topology_translation->my_host_name) {
            endpoints.push_back(connection.endpoint_b());
        }
    }
    log_info(tt::LogAlways, "Found {} endpoints to monitor in factory system descriptor", endpoints.size());
    if (endpoints.size() == 0 && fsd.eth_connections().connection().size() > 0) {
        log_warning(tt::LogAlways, "Found 0 endpoints to monitor despite {} existing in factory system descriptor. Please check that the hostname in the FSD file matches the actual system hostname of this machine.", fsd.eth_connections().connection().size());
    }

    // For each, create a metric
    for (const auto& endpoint : endpoints) {
        uint32_t channel = endpoint.chan_id();
        tt::tt_metal::ASICLocation asic_location = tt::tt_metal::ASICLocation(endpoint.asic_location());
        tt::tt_metal::TrayID tray_id = tt::tt_metal::TrayID(endpoint.tray_id());
        std::optional<tt::ChipId> chip_id_optional =
            topology_translation->get_local_chip_id_for_asic_location_and_tray(asic_location, tray_id);
        TT_FATAL(
            chip_id_optional.has_value(),
            "Unable to map ASIC location {} and tray {} to a chip ID",
            *asic_location,
            *tray_id);
        tt::ChipId chip_id = chip_id_optional.value();

        log_info(
            tt::LogAlways,
            "Creating Ethernet metrics for tray_id={}, asic_location={}, channel={}, chip_id={}...",
            *tray_id,
            *asic_location,
            channel,
            chip_id);
        bool_metrics.push_back(
            std::make_unique<EthernetEndpointUpMetric>(tray_id, asic_location, chip_id, channel, hal));
        uint_metrics.push_back(
            std::make_unique<EthernetRetrainCountMetric>(tray_id, asic_location, chip_id, channel, cluster, hal));
        auto bandwidth_reader = std::make_shared<FabricBandwidthTelemetryReader>(
            tray_id, asic_location, chip_id, channel, cluster, hal);
        double_metrics.push_back(
            std::make_unique<FabricBandwidthMetric>(tray_id, asic_location, chip_id, channel, bandwidth_reader));
        uint_metrics.push_back(
            std::make_unique<FabricWordsSentMetric>(tray_id, asic_location, chip_id, channel, bandwidth_reader));
        uint_metrics.push_back(
            std::make_unique<FabricPacketsSentMetric>(tray_id, asic_location, chip_id, channel, bandwidth_reader));
        if (hal->get_arch() == tt::ARCH::WORMHOLE_B0) {
            // These are available only on Wormhole
            uint_metrics.push_back(
                std::make_unique<EthernetCRCErrorCountMetric>(tray_id, asic_location, chip_id, channel, cluster, hal));
            uint_metrics.push_back(std::make_unique<EthernetCorrectedCodewordCountMetric>(
                tray_id, asic_location, chip_id, channel, cluster, hal));
            uint_metrics.push_back(std::make_unique<EthernetUncorrectedCodewordCountMetric>(
                tray_id, asic_location, chip_id, channel, cluster, hal));
        }
        //TODO: this does not appear to be working as expected, needs more debugging and development work
        // uint_metrics.push_back(
        //     std::make_unique<FabricHeartbeatMetric>(tray_id, asic_location, chip_id, channel, cluster, hal));
    }

    log_info(tt::LogAlways, "Created Ethernet metrics");
}

static std::vector<std::string> endpoint_telemetry_path(
    tt::tt_metal::TrayID tray_id, tt::tt_metal::ASICLocation asic_location, uint32_t channel, const char* metric_name) {
    // Create path in format: tray{n}/chip{m}/channel{l}/metric_name
    return {
        "tray" + std::to_string(*tray_id),
        "chip" + std::to_string(*asic_location),
        "channel" + std::to_string(channel),
        metric_name};
}

/**************************************************************************************************
 EthernetEndpointUpMetric

 Whether link is up.
**************************************************************************************************/

EthernetEndpointUpMetric::EthernetEndpointUpMetric(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    tt::ChipId chip_id,
    uint32_t channel,
    const std::unique_ptr<tt::tt_metal::Hal>& hal) :
    BoolMetric(),
    tray_id_(tray_id),
    asic_location_(asic_location),
    chip_id_(chip_id),
    channel_(channel),
    last_force_refresh_time_(std::chrono::steady_clock::time_point::min()),
    link_up_addr_(hal->get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::LINK_UP)) {}

const std::vector<std::string> EthernetEndpointUpMetric::telemetry_path() const {
    return endpoint_telemetry_path(tray_id_, asic_location_, channel_, "linkIsUp");
}

void EthernetEndpointUpMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    // Check if enough time has elapsed since last force refresh
    bool should_force_refresh = (start_of_update_cycle - last_force_refresh_time_) >= FORCE_REFRESH_LINK_STATUS_TIMEOUT;

    // If we're forcing a refresh, update the timestamp
    if (should_force_refresh) {
        last_force_refresh_time_ = start_of_update_cycle;
    }

    bool is_up_now = is_ethernet_endpoint_up(cluster, chip_id_, channel_, link_up_addr_, should_force_refresh);
    bool is_up_old = value_;
    changed_since_transmission_ = is_up_now != is_up_old;
    value_ = is_up_now;
    set_timestamp_now();
}

/**************************************************************************************************
 EthernetCRCErrorCountMetric

 Number of CRC errors encountered.
**************************************************************************************************/

EthernetCRCErrorCountMetric::EthernetCRCErrorCountMetric(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    tt::ChipId chip_id,
    uint32_t channel,
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const std::unique_ptr<tt::tt_metal::Hal>& hal) :
    UIntMetric(), tray_id_(tray_id), asic_location_(asic_location), chip_id_(chip_id), channel_(channel) {
    value_ = 0;
    tt::umd::TTDevice* device = cluster->get_tt_device(chip_id);
    TT_FATAL(device->get_arch() == tt::ARCH::WORMHOLE_B0, "Metric {} available only on Wormhole", __func__);
    ethernet_core_ = cluster->get_soc_descriptor(chip_id).get_eth_core_for_channel(channel, tt::CoordSystem::LOGICAL);
    crc_addr_ = hal->get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::CRC_ERR);
}

const std::vector<std::string> EthernetCRCErrorCountMetric::telemetry_path() const {
    return endpoint_telemetry_path(tray_id_, asic_location_, channel_, "crcErrorCount");
}

void EthernetCRCErrorCountMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    uint32_t crc_error_val = 0;
    cluster->read_from_device(&crc_error_val, chip_id_, ethernet_core_, crc_addr_, sizeof(uint32_t));
    value_ = uint64_t(crc_error_val);
    set_timestamp_now();
}

/**************************************************************************************************
 EthernetRetrainCountMetric

 Number of CRC retrains that occurred.
**************************************************************************************************/

EthernetRetrainCountMetric::EthernetRetrainCountMetric(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    tt::ChipId chip_id,
    uint32_t channel,
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const std::unique_ptr<tt::tt_metal::Hal>& hal) :
    UIntMetric(), tray_id_(tray_id), asic_location_(asic_location), chip_id_(chip_id), channel_(channel) {
    value_ = 0;
    ethernet_core_ = cluster->get_soc_descriptor(chip_id).get_eth_core_for_channel(channel, tt::CoordSystem::LOGICAL);
    retrain_count_addr_ = hal->get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::RETRAIN_COUNT);
}

const std::vector<std::string> EthernetRetrainCountMetric::telemetry_path() const {
    return endpoint_telemetry_path(tray_id_, asic_location_, channel_, "retrainCount");
}

void EthernetRetrainCountMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    uint32_t data = 0;
    cluster->read_from_device(&data, chip_id_, ethernet_core_, retrain_count_addr_, sizeof(uint32_t));
    value_ = uint64_t(data);
    set_timestamp_now();
}

/**************************************************************************************************
 EthernetCorrectedCodewordCountMetric

 Number of codeword corrections.
**************************************************************************************************/

EthernetCorrectedCodewordCountMetric::EthernetCorrectedCodewordCountMetric(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    tt::ChipId chip_id,
    uint32_t channel,
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const std::unique_ptr<tt::tt_metal::Hal>& hal) :
    UIntMetric(), tray_id_(tray_id), asic_location_(asic_location), chip_id_(chip_id), channel_(channel) {
    value_ = 0;
    tt::umd::TTDevice* device = cluster->get_tt_device(chip_id);
    TT_FATAL(device->get_arch() == tt::ARCH::WORMHOLE_B0, "Metric {} available only on Wormhole", __func__);
    ethernet_core_ = cluster->get_soc_descriptor(chip_id).get_eth_core_for_channel(channel, tt::CoordSystem::LOGICAL);
    corr_addr_ = hal->get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::CORR_CW);
}

const std::vector<std::string> EthernetCorrectedCodewordCountMetric::telemetry_path() const {
    return endpoint_telemetry_path(tray_id_, asic_location_, channel_, "codewordCount");
}

void EthernetCorrectedCodewordCountMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    uint32_t hi = 0;
    uint32_t lo = 0;
    cluster->read_from_device(&hi, chip_id_, ethernet_core_, corr_addr_ + 0, sizeof(uint32_t));
    cluster->read_from_device(&lo, chip_id_, ethernet_core_, corr_addr_ + 4, sizeof(uint32_t));
    value_ = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
    set_timestamp_now();
}

/**************************************************************************************************
 EthernetUncorrectedCodewordCountMetric

 Number of uncorrected codewords.
**************************************************************************************************/

EthernetUncorrectedCodewordCountMetric::EthernetUncorrectedCodewordCountMetric(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    tt::ChipId chip_id,
    uint32_t channel,
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const std::unique_ptr<tt::tt_metal::Hal>& hal) :
    UIntMetric(), tray_id_(tray_id), asic_location_(asic_location), chip_id_(chip_id), channel_(channel) {
    value_ = 0;
    tt::umd::TTDevice* device = cluster->get_tt_device(chip_id);
    TT_FATAL(device->get_arch() == tt::ARCH::WORMHOLE_B0, "Metric {} available only on Wormhole", __func__);
    ethernet_core_ = cluster->get_soc_descriptor(chip_id).get_eth_core_for_channel(channel, tt::CoordSystem::LOGICAL);
    uncorr_addr_ = hal->get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNCORR_CW);
}

const std::vector<std::string> EthernetUncorrectedCodewordCountMetric::telemetry_path() const {
    return endpoint_telemetry_path(tray_id_, asic_location_, channel_, "uncorrectedCodewordCount");
}

void EthernetUncorrectedCodewordCountMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    uint32_t hi = 0;
    uint32_t lo = 0;
    cluster->read_from_device(&hi, chip_id_, ethernet_core_, uncorr_addr_ + 0, sizeof(uint32_t));
    cluster->read_from_device(&lo, chip_id_, ethernet_core_, uncorr_addr_ + 4, sizeof(uint32_t));
    value_ = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
    set_timestamp_now();
}

/**************************************************************************************************
 FabricBandwidthMetric

 Bandwidth telemetry.
**************************************************************************************************/

static double calc_bw_bytes_per_cycle(uint32_t total_words, uint64_t cycles) {
    constexpr uint32_t bytes_per_eth_word = 16;
    return (total_words * bytes_per_eth_word) / static_cast<double>(cycles);
}

FabricBandwidthMetric::FabricBandwidthMetric(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    tt::ChipId chip_id,
    uint32_t channel,
    std::shared_ptr<FabricBandwidthTelemetryReader> telemetry_reader) :
    DoubleMetric(),
    tray_id_(tray_id),
    asic_location_(asic_location),
    chip_id_(chip_id),
    channel_(channel),
    telemetry_reader_(telemetry_reader) {
    value_ = 0.0;
}

const std::vector<std::string> FabricBandwidthMetric::telemetry_path() const {
    return endpoint_telemetry_path(tray_id_, asic_location_, channel_, "fabricBandwidth");
}

void FabricBandwidthMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    // Get telemetry data from the shared reader
    const LowResolutionBandwidthTelemetryResult& tel = telemetry_reader_->get_telemetry(cluster, start_of_update_cycle);
    
    // Calculate bandwidth
    double old_value = value_;
    uint64_t cycles = tel.duration.full;
    if (cycles > 0) {
        // Get device frequency in MHz
        uint32_t freq_mhz = cluster->get_tt_device(chip_id_)->get_clock();
        double freq_ghz = double(freq_mhz) / 1000.0;
        
        double bytes_per_cycle = calc_bw_bytes_per_cycle(tel.num_words_sent, cycles);
        value_ = bytes_per_cycle * freq_ghz;  // GB/s
    } else {
        value_ = 0.0;
    }
    
    // Mark as changed if value differs from previous reading
    changed_since_transmission_ = (value_ != old_value);
    
    timestamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

/**************************************************************************************************
 FabricWordsSentMetric

 Number of words sent over fabric.
**************************************************************************************************/

FabricWordsSentMetric::FabricWordsSentMetric(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    tt::ChipId chip_id,
    uint32_t channel,
    std::shared_ptr<FabricBandwidthTelemetryReader> telemetry_reader) :
    UIntMetric(),
    tray_id_(tray_id),
    asic_location_(asic_location),
    chip_id_(chip_id),
    channel_(channel),
    telemetry_reader_(telemetry_reader) {
    value_ = 0;
}

const std::vector<std::string> FabricWordsSentMetric::telemetry_path() const {
    return endpoint_telemetry_path(tray_id_, asic_location_, channel_, "fabricWordsSent");
}

void FabricWordsSentMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    // Get telemetry data from the shared reader
    const LowResolutionBandwidthTelemetryResult& tel = telemetry_reader_->get_telemetry(cluster, start_of_update_cycle);
    
    uint64_t old_value = value_;
    value_ = tel.num_words_sent;
    
    // Mark as changed if value differs from previous reading
    changed_since_transmission_ = (value_ != old_value);
    
    timestamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

/**************************************************************************************************
 FabricPacketsSentMetric

 Number of packets sent over fabric.
**************************************************************************************************/

FabricPacketsSentMetric::FabricPacketsSentMetric(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    tt::ChipId chip_id,
    uint32_t channel,
    std::shared_ptr<FabricBandwidthTelemetryReader> telemetry_reader) :
    UIntMetric(),
    tray_id_(tray_id),
    asic_location_(asic_location),
    chip_id_(chip_id),
    channel_(channel),
    telemetry_reader_(telemetry_reader) {
    value_ = 0;
}

const std::vector<std::string> FabricPacketsSentMetric::telemetry_path() const {
    return endpoint_telemetry_path(tray_id_, asic_location_, channel_, "fabricPacketsSent");
}

void FabricPacketsSentMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    // Get telemetry data from the shared reader
    const LowResolutionBandwidthTelemetryResult& tel = telemetry_reader_->get_telemetry(cluster, start_of_update_cycle);
    
    uint64_t old_value = value_;
    value_ = tel.num_packets_sent;
    
    // Mark as changed if value differs from previous reading
    changed_since_transmission_ = (value_ != old_value);
    
    timestamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

/**************************************************************************************************
 FabricHeartbeatMetric

 Fabric heartbeat count.
**************************************************************************************************/

FabricHeartbeatMetric::FabricHeartbeatMetric(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    tt::ChipId chip_id,
    uint32_t channel,
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const std::unique_ptr<tt::tt_metal::Hal>& hal) :
    UIntMetric(), tray_id_(tray_id), asic_location_(asic_location), chip_id_(chip_id), channel_(channel) {
    value_ = 0;
    ethernet_core_ = cluster->get_soc_descriptor(chip_id).get_eth_core_for_channel(channel, tt::CoordSystem::LOGICAL);
    heartbeat_addr_ = hal->get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::FABRIC_HEARTBEAT);
}

const std::vector<std::string> FabricHeartbeatMetric::telemetry_path() const {
    return endpoint_telemetry_path(tray_id_, asic_location_, channel_, "fabricHeartbeat");
}

void FabricHeartbeatMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    cluster->read_from_device(&value_, chip_id_, ethernet_core_, heartbeat_addr_, sizeof(uint64_t));
    timestamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}