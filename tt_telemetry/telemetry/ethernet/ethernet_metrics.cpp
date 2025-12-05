// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>

#include "protobuf/factory_system_descriptor.pb.h"

#include <telemetry/ethernet/ethernet_metrics.hpp>
#include <telemetry/ethernet/ethernet_helpers.hpp>
#include <telemetry/ethernet/fabric_telemetry_reader.hpp>
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
        bool_metrics.push_back(std::make_unique<EthernetEndpointUpMetric>(
            tray_id, asic_location, chip_id, channel, hal, topology_translation));
        uint_metrics.push_back(std::make_unique<EthernetRetrainCountMetric>(
            tray_id, asic_location, chip_id, channel, cluster, hal, topology_translation));
        if (hal->get_arch() == tt::ARCH::WORMHOLE_B0) {
            // These are available only on Wormhole
            uint_metrics.push_back(std::make_unique<EthernetCRCErrorCountMetric>(
                tray_id, asic_location, chip_id, channel, cluster, hal, topology_translation));
            uint_metrics.push_back(std::make_unique<EthernetCorrectedCodewordCountMetric>(
                tray_id, asic_location, chip_id, channel, cluster, hal, topology_translation));
            uint_metrics.push_back(std::make_unique<EthernetUncorrectedCodewordCountMetric>(
                tray_id, asic_location, chip_id, channel, cluster, hal, topology_translation));
        }

        // Create fabric telemetry metrics (available on Wormhole and Blackhole)
        // These metrics read from device using Daiki's fabric telemetry infrastructure
        // Controlled by TT_METAL_FABRIC_TELEMETRY=1 environment variable
        if (hal->get_arch() == tt::ARCH::WORMHOLE_B0 || hal->get_arch() == tt::ARCH::BLACKHOLE) {
            // Create a shared telemetry reader per chip to enable caching across multiple metrics
            auto telemetry_reader = std::make_shared<FabricTelemetryReader>(chip_id);

            // Raw bandwidth counters
            uint_metrics.push_back(std::make_unique<FabricTxWordsMetric>(
                tray_id, asic_location, channel, telemetry_reader, topology_translation));
            uint_metrics.push_back(std::make_unique<FabricRxWordsMetric>(
                tray_id, asic_location, channel, telemetry_reader, topology_translation));
            uint_metrics.push_back(std::make_unique<FabricTxPacketsMetric>(
                tray_id, asic_location, channel, telemetry_reader, topology_translation));
            uint_metrics.push_back(std::make_unique<FabricRxPacketsMetric>(
                tray_id, asic_location, channel, telemetry_reader, topology_translation));
            uint_metrics.push_back(std::make_unique<FabricTxElapsedCyclesMetric>(
                tray_id, asic_location, channel, telemetry_reader, topology_translation));
            uint_metrics.push_back(std::make_unique<FabricRxElapsedCyclesMetric>(
                tray_id, asic_location, channel, telemetry_reader, topology_translation));
            uint_metrics.push_back(std::make_unique<FabricTxActiveCyclesMetric>(
                tray_id, asic_location, channel, telemetry_reader, topology_translation));
            uint_metrics.push_back(std::make_unique<FabricRxActiveCyclesMetric>(
                tray_id, asic_location, channel, telemetry_reader, topology_translation));

            // ERISC-specific metrics (heartbeat and router state)
            // Number of ERISC cores depends on architecture
            size_t num_erisc_cores = (hal->get_arch() == tt::ARCH::BLACKHOLE) ? 2 : 1;
            for (size_t erisc_core = 0; erisc_core < num_erisc_cores; erisc_core++) {
                uint_metrics.push_back(std::make_unique<FabricTxHeartbeatMetric>(
                    tray_id, asic_location, channel, erisc_core, telemetry_reader, topology_translation));
                uint_metrics.push_back(std::make_unique<FabricRxHeartbeatMetric>(
                    tray_id, asic_location, channel, erisc_core, telemetry_reader, topology_translation));
                uint_metrics.push_back(std::make_unique<FabricRouterStateMetric>(
                    tray_id, asic_location, channel, erisc_core, telemetry_reader, topology_translation));
            }
        }
    }

    log_info(tt::LogAlways, "Created Ethernet metrics");
}

// Helper to get physical link info for Ethernet metrics
static std::optional<PhysicalLinkInfo> get_physical_link_info_for_endpoint(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    uint32_t channel,
    const std::unique_ptr<TopologyHelper>& topology_helper) {
    if (!topology_helper) {
        return std::nullopt;
    }

    EthernetEndpoint endpoint{tray_id, asic_location, channel};
    return topology_helper->get_physical_link_info(endpoint);
}

// Helper to construct hierarchical telemetry path for Ethernet endpoint metrics
static std::vector<std::string> build_ethernet_endpoint_path(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    uint32_t channel,
    const std::string& metric_name) {
    return {
        "tray" + std::to_string(*tray_id),
        "chip" + std::to_string(*asic_location),
        "channel" + std::to_string(channel),
        metric_name};
}

// Helper to construct labels for Ethernet metrics (base + optional topology info)
static std::unordered_map<std::string, std::string> build_ethernet_labels(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    uint32_t channel,
    const std::optional<PhysicalLinkInfo>& link_info) {
    std::unordered_map<std::string, std::string> result;
    result["tray"] = std::to_string(*tray_id);
    result["chip"] = std::to_string(*asic_location);
    result["channel"] = std::to_string(channel);

    if (link_info.has_value()) {
        result["port_type"] = std::to_string(static_cast<int>(link_info->port_type));
        result["port_id"] = std::to_string(*link_info->port_id);

        if (link_info->remote_endpoint.has_value()) {
            const auto& remote = link_info->remote_endpoint.value();
            result["remote_hostname"] = remote.hostname;
            result["remote_tray"] = std::to_string(*remote.tray);
            result["remote_chip"] = std::to_string(*remote.asic);
            result["remote_channel"] = std::to_string(remote.channel);
            result["remote_rack"] = std::to_string(remote.rack);
        }
    }

    return result;
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
    const std::unique_ptr<tt::tt_metal::Hal>& hal,
    const std::unique_ptr<TopologyHelper>& topology_helper) :
    BoolMetric(),
    tray_id_(tray_id),
    asic_location_(asic_location),
    chip_id_(chip_id),
    channel_(channel),
    last_force_refresh_time_(std::chrono::steady_clock::time_point::min()),
    link_up_addr_(
        hal->get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::LINK_UP)),
    link_info_(get_physical_link_info_for_endpoint(tray_id, asic_location, channel, topology_helper)) {}

const std::vector<std::string> EthernetEndpointUpMetric::telemetry_path() const {
    return build_ethernet_endpoint_path(tray_id_, asic_location_, channel_, "linkIsUp");
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

std::unordered_map<std::string, std::string> EthernetEndpointUpMetric::labels() const {
    return build_ethernet_labels(tray_id_, asic_location_, channel_, link_info_);
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
    const std::unique_ptr<tt::tt_metal::Hal>& hal,
    const std::unique_ptr<TopologyHelper>& topology_helper) :
    UIntMetric(), tray_id_(tray_id), asic_location_(asic_location), chip_id_(chip_id), channel_(channel) {
    value_ = 0;
    tt::umd::TTDevice* device = cluster->get_tt_device(chip_id);
    TT_FATAL(device->get_arch() == tt::ARCH::WORMHOLE_B0, "Metric {} available only on Wormhole", __func__);
    ethernet_core_ = cluster->get_soc_descriptor(chip_id).get_eth_core_for_channel(channel, tt::CoordSystem::LOGICAL);
    crc_addr_ = hal->get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::CRC_ERR);
    link_info_ = get_physical_link_info_for_endpoint(tray_id, asic_location, channel, topology_helper);
}

const std::vector<std::string> EthernetCRCErrorCountMetric::telemetry_path() const {
    return build_ethernet_endpoint_path(tray_id_, asic_location_, channel_, "crcErrorCount");
}

void EthernetCRCErrorCountMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    uint32_t crc_error_val = 0;
    cluster->read_from_device(&crc_error_val, chip_id_, ethernet_core_, crc_addr_, sizeof(uint32_t));
    value_ = uint64_t(crc_error_val);
    set_timestamp_now();
}

std::unordered_map<std::string, std::string> EthernetCRCErrorCountMetric::labels() const {
    return build_ethernet_labels(tray_id_, asic_location_, channel_, link_info_);
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
    const std::unique_ptr<tt::tt_metal::Hal>& hal,
    const std::unique_ptr<TopologyHelper>& topology_helper) :
    UIntMetric(), tray_id_(tray_id), asic_location_(asic_location), chip_id_(chip_id), channel_(channel) {
    value_ = 0;
    ethernet_core_ = cluster->get_soc_descriptor(chip_id).get_eth_core_for_channel(channel, tt::CoordSystem::LOGICAL);
    retrain_count_addr_ = hal->get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::RETRAIN_COUNT);
    link_info_ = get_physical_link_info_for_endpoint(tray_id, asic_location, channel, topology_helper);
}

const std::vector<std::string> EthernetRetrainCountMetric::telemetry_path() const {
    return build_ethernet_endpoint_path(tray_id_, asic_location_, channel_, "retrainCount");
}

void EthernetRetrainCountMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    uint32_t data = 0;
    cluster->read_from_device(&data, chip_id_, ethernet_core_, retrain_count_addr_, sizeof(uint32_t));
    value_ = uint64_t(data);
    set_timestamp_now();
}

std::unordered_map<std::string, std::string> EthernetRetrainCountMetric::labels() const {
    return build_ethernet_labels(tray_id_, asic_location_, channel_, link_info_);
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
    const std::unique_ptr<tt::tt_metal::Hal>& hal,
    const std::unique_ptr<TopologyHelper>& topology_helper) :
    UIntMetric(), tray_id_(tray_id), asic_location_(asic_location), chip_id_(chip_id), channel_(channel) {
    value_ = 0;
    tt::umd::TTDevice* device = cluster->get_tt_device(chip_id);
    TT_FATAL(device->get_arch() == tt::ARCH::WORMHOLE_B0, "Metric {} available only on Wormhole", __func__);
    ethernet_core_ = cluster->get_soc_descriptor(chip_id).get_eth_core_for_channel(channel, tt::CoordSystem::LOGICAL);
    corr_addr_ = hal->get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::CORR_CW);
    link_info_ = get_physical_link_info_for_endpoint(tray_id, asic_location, channel, topology_helper);
}

const std::vector<std::string> EthernetCorrectedCodewordCountMetric::telemetry_path() const {
    return build_ethernet_endpoint_path(tray_id_, asic_location_, channel_, "correctedCodewordCount");
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

std::unordered_map<std::string, std::string> EthernetCorrectedCodewordCountMetric::labels() const {
    return build_ethernet_labels(tray_id_, asic_location_, channel_, link_info_);
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
    const std::unique_ptr<tt::tt_metal::Hal>& hal,
    const std::unique_ptr<TopologyHelper>& topology_helper) :
    UIntMetric(), tray_id_(tray_id), asic_location_(asic_location), chip_id_(chip_id), channel_(channel) {
    value_ = 0;
    tt::umd::TTDevice* device = cluster->get_tt_device(chip_id);
    TT_FATAL(device->get_arch() == tt::ARCH::WORMHOLE_B0, "Metric {} available only on Wormhole", __func__);
    ethernet_core_ = cluster->get_soc_descriptor(chip_id).get_eth_core_for_channel(channel, tt::CoordSystem::LOGICAL);
    uncorr_addr_ = hal->get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNCORR_CW);
    link_info_ = get_physical_link_info_for_endpoint(tray_id, asic_location, channel, topology_helper);
}

const std::vector<std::string> EthernetUncorrectedCodewordCountMetric::telemetry_path() const {
    return build_ethernet_endpoint_path(tray_id_, asic_location_, channel_, "uncorrectedCodewordCount");
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

std::unordered_map<std::string, std::string> EthernetUncorrectedCodewordCountMetric::labels() const {
    return build_ethernet_labels(tray_id_, asic_location_, channel_, link_info_);
}

/**************************************************************************************************
 Fabric Telemetry Metrics - Helper Functions
**************************************************************************************************/

// Helper to build telemetry path for fabric metrics (per channel)
static std::vector<std::string> build_fabric_endpoint_path(
    tt::tt_metal::TrayID tray_id, tt::tt_metal::ASICLocation asic_location, uint32_t channel, const char* metric_name) {
    return {
        "tray" + std::to_string(*tray_id),
        "chip" + std::to_string(*asic_location),
        "channel" + std::to_string(channel),
        "fabric",
        metric_name};
}

// Helper to build telemetry path for fabric metrics (per ERISC)
static std::vector<std::string> build_fabric_erisc_path(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    uint32_t channel,
    size_t erisc_core,
    const char* metric_name) {
    return {
        "tray" + std::to_string(*tray_id),
        "chip" + std::to_string(*asic_location),
        "channel" + std::to_string(channel),
        "fabric",
        "erisc" + std::to_string(erisc_core),
        metric_name};
}

// Helper to find the telemetry sample for a specific channel
static const tt::tt_fabric::FabricTelemetrySample* find_sample_for_channel(
    const std::vector<tt::tt_fabric::FabricTelemetrySample>& samples, uint32_t channel) {
    for (const auto& sample : samples) {
        if (sample.channel_id == channel) {
            return &sample;
        }
    }
    return nullptr;
}

/**************************************************************************************************
 FabricTxWordsMetric
**************************************************************************************************/

FabricTxWordsMetric::FabricTxWordsMetric(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    uint32_t channel,
    std::shared_ptr<FabricTelemetryReader> telemetry_reader,
    const std::unique_ptr<TopologyHelper>& topology_helper) :
    UIntMetric(),
    tray_id_(tray_id),
    asic_location_(asic_location),
    channel_(channel),
    telemetry_reader_(telemetry_reader) {
    value_ = 0;
    link_info_ = get_physical_link_info_for_endpoint(tray_id, asic_location, channel, topology_helper);
}

const std::vector<std::string> FabricTxWordsMetric::telemetry_path() const {
    return build_fabric_endpoint_path(tray_id_, asic_location_, channel_, "txWords");
}

void FabricTxWordsMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    const auto& samples = telemetry_reader_->get_fabric_telemetry(start_of_update_cycle);
    const auto* sample = find_sample_for_channel(samples, channel_);
    if (sample && sample->snapshot.dynamic_info.has_value()) {
        uint64_t new_value = sample->snapshot.dynamic_info->tx_bandwidth.words_sent;
        changed_since_transmission_ = new_value != value_;
        value_ = new_value;
    }
    set_timestamp_now();
}

std::unordered_map<std::string, std::string> FabricTxWordsMetric::labels() const {
    return build_ethernet_labels(tray_id_, asic_location_, channel_, link_info_);
}

/**************************************************************************************************
 FabricRxWordsMetric
**************************************************************************************************/

FabricRxWordsMetric::FabricRxWordsMetric(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    uint32_t channel,
    std::shared_ptr<FabricTelemetryReader> telemetry_reader,
    const std::unique_ptr<TopologyHelper>& topology_helper) :
    UIntMetric(),
    tray_id_(tray_id),
    asic_location_(asic_location),
    channel_(channel),
    telemetry_reader_(telemetry_reader) {
    value_ = 0;
    link_info_ = get_physical_link_info_for_endpoint(tray_id, asic_location, channel, topology_helper);
}

const std::vector<std::string> FabricRxWordsMetric::telemetry_path() const {
    return build_fabric_endpoint_path(tray_id_, asic_location_, channel_, "rxWords");
}

void FabricRxWordsMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    const auto& samples = telemetry_reader_->get_fabric_telemetry(start_of_update_cycle);
    const auto* sample = find_sample_for_channel(samples, channel_);
    if (sample && sample->snapshot.dynamic_info.has_value()) {
        uint64_t new_value = sample->snapshot.dynamic_info->rx_bandwidth.words_sent;
        changed_since_transmission_ = new_value != value_;
        value_ = new_value;
    }
    set_timestamp_now();
}

std::unordered_map<std::string, std::string> FabricRxWordsMetric::labels() const {
    return build_ethernet_labels(tray_id_, asic_location_, channel_, link_info_);
}

/**************************************************************************************************
 FabricTxPacketsMetric
**************************************************************************************************/

FabricTxPacketsMetric::FabricTxPacketsMetric(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    uint32_t channel,
    std::shared_ptr<FabricTelemetryReader> telemetry_reader,
    const std::unique_ptr<TopologyHelper>& topology_helper) :
    UIntMetric(),
    tray_id_(tray_id),
    asic_location_(asic_location),
    channel_(channel),
    telemetry_reader_(telemetry_reader) {
    value_ = 0;
    link_info_ = get_physical_link_info_for_endpoint(tray_id, asic_location, channel, topology_helper);
}

const std::vector<std::string> FabricTxPacketsMetric::telemetry_path() const {
    return build_fabric_endpoint_path(tray_id_, asic_location_, channel_, "txPackets");
}

void FabricTxPacketsMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    const auto& samples = telemetry_reader_->get_fabric_telemetry(start_of_update_cycle);
    const auto* sample = find_sample_for_channel(samples, channel_);
    if (sample && sample->snapshot.dynamic_info.has_value()) {
        uint64_t new_value = sample->snapshot.dynamic_info->tx_bandwidth.packets_sent;
        changed_since_transmission_ = new_value != value_;
        value_ = new_value;
    }
    set_timestamp_now();
}

std::unordered_map<std::string, std::string> FabricTxPacketsMetric::labels() const {
    return build_ethernet_labels(tray_id_, asic_location_, channel_, link_info_);
}

/**************************************************************************************************
 FabricRxPacketsMetric
**************************************************************************************************/

FabricRxPacketsMetric::FabricRxPacketsMetric(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    uint32_t channel,
    std::shared_ptr<FabricTelemetryReader> telemetry_reader,
    const std::unique_ptr<TopologyHelper>& topology_helper) :
    UIntMetric(),
    tray_id_(tray_id),
    asic_location_(asic_location),
    channel_(channel),
    telemetry_reader_(telemetry_reader) {
    value_ = 0;
    link_info_ = get_physical_link_info_for_endpoint(tray_id, asic_location, channel, topology_helper);
}

const std::vector<std::string> FabricRxPacketsMetric::telemetry_path() const {
    return build_fabric_endpoint_path(tray_id_, asic_location_, channel_, "rxPackets");
}

void FabricRxPacketsMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    const auto& samples = telemetry_reader_->get_fabric_telemetry(start_of_update_cycle);
    const auto* sample = find_sample_for_channel(samples, channel_);
    if (sample && sample->snapshot.dynamic_info.has_value()) {
        uint64_t new_value = sample->snapshot.dynamic_info->rx_bandwidth.packets_sent;
        changed_since_transmission_ = new_value != value_;
        value_ = new_value;
    }
    set_timestamp_now();
}

std::unordered_map<std::string, std::string> FabricRxPacketsMetric::labels() const {
    return build_ethernet_labels(tray_id_, asic_location_, channel_, link_info_);
}

/**************************************************************************************************
 FabricTxElapsedCyclesMetric
**************************************************************************************************/

FabricTxElapsedCyclesMetric::FabricTxElapsedCyclesMetric(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    uint32_t channel,
    std::shared_ptr<FabricTelemetryReader> telemetry_reader,
    const std::unique_ptr<TopologyHelper>& topology_helper) :
    UIntMetric(),
    tray_id_(tray_id),
    asic_location_(asic_location),
    channel_(channel),
    telemetry_reader_(telemetry_reader) {
    value_ = 0;
    link_info_ = get_physical_link_info_for_endpoint(tray_id, asic_location, channel, topology_helper);
}

const std::vector<std::string> FabricTxElapsedCyclesMetric::telemetry_path() const {
    return build_fabric_endpoint_path(tray_id_, asic_location_, channel_, "txElapsedCycles");
}

void FabricTxElapsedCyclesMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    const auto& samples = telemetry_reader_->get_fabric_telemetry(start_of_update_cycle);
    const auto* sample = find_sample_for_channel(samples, channel_);
    if (sample && sample->snapshot.dynamic_info.has_value()) {
        uint64_t new_value = sample->snapshot.dynamic_info->tx_bandwidth.elapsed_cycles;
        changed_since_transmission_ = new_value != value_;
        value_ = new_value;
    }
    set_timestamp_now();
}

std::unordered_map<std::string, std::string> FabricTxElapsedCyclesMetric::labels() const {
    return build_ethernet_labels(tray_id_, asic_location_, channel_, link_info_);
}

/**************************************************************************************************
 FabricRxElapsedCyclesMetric
**************************************************************************************************/

FabricRxElapsedCyclesMetric::FabricRxElapsedCyclesMetric(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    uint32_t channel,
    std::shared_ptr<FabricTelemetryReader> telemetry_reader,
    const std::unique_ptr<TopologyHelper>& topology_helper) :
    UIntMetric(),
    tray_id_(tray_id),
    asic_location_(asic_location),
    channel_(channel),
    telemetry_reader_(telemetry_reader) {
    value_ = 0;
    link_info_ = get_physical_link_info_for_endpoint(tray_id, asic_location, channel, topology_helper);
}

const std::vector<std::string> FabricRxElapsedCyclesMetric::telemetry_path() const {
    return build_fabric_endpoint_path(tray_id_, asic_location_, channel_, "rxElapsedCycles");
}

void FabricRxElapsedCyclesMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    const auto& samples = telemetry_reader_->get_fabric_telemetry(start_of_update_cycle);
    const auto* sample = find_sample_for_channel(samples, channel_);
    if (sample && sample->snapshot.dynamic_info.has_value()) {
        uint64_t new_value = sample->snapshot.dynamic_info->rx_bandwidth.elapsed_cycles;
        changed_since_transmission_ = new_value != value_;
        value_ = new_value;
    }
    set_timestamp_now();
}

std::unordered_map<std::string, std::string> FabricRxElapsedCyclesMetric::labels() const {
    return build_ethernet_labels(tray_id_, asic_location_, channel_, link_info_);
}

/**************************************************************************************************
 FabricTxActiveCyclesMetric
**************************************************************************************************/

FabricTxActiveCyclesMetric::FabricTxActiveCyclesMetric(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    uint32_t channel,
    std::shared_ptr<FabricTelemetryReader> telemetry_reader,
    const std::unique_ptr<TopologyHelper>& topology_helper) :
    UIntMetric(),
    tray_id_(tray_id),
    asic_location_(asic_location),
    channel_(channel),
    telemetry_reader_(telemetry_reader) {
    value_ = 0;
    link_info_ = get_physical_link_info_for_endpoint(tray_id, asic_location, channel, topology_helper);
}

const std::vector<std::string> FabricTxActiveCyclesMetric::telemetry_path() const {
    return build_fabric_endpoint_path(tray_id_, asic_location_, channel_, "txActiveCycles");
}

void FabricTxActiveCyclesMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    const auto& samples = telemetry_reader_->get_fabric_telemetry(start_of_update_cycle);
    const auto* sample = find_sample_for_channel(samples, channel_);
    if (sample && sample->snapshot.dynamic_info.has_value()) {
        uint64_t new_value = sample->snapshot.dynamic_info->tx_bandwidth.elapsed_active_cycles;
        changed_since_transmission_ = new_value != value_;
        value_ = new_value;
    }
    set_timestamp_now();
}

std::unordered_map<std::string, std::string> FabricTxActiveCyclesMetric::labels() const {
    return build_ethernet_labels(tray_id_, asic_location_, channel_, link_info_);
}

/**************************************************************************************************
 FabricRxActiveCyclesMetric
**************************************************************************************************/

FabricRxActiveCyclesMetric::FabricRxActiveCyclesMetric(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    uint32_t channel,
    std::shared_ptr<FabricTelemetryReader> telemetry_reader,
    const std::unique_ptr<TopologyHelper>& topology_helper) :
    UIntMetric(),
    tray_id_(tray_id),
    asic_location_(asic_location),
    channel_(channel),
    telemetry_reader_(telemetry_reader) {
    value_ = 0;
    link_info_ = get_physical_link_info_for_endpoint(tray_id, asic_location, channel, topology_helper);
}

const std::vector<std::string> FabricRxActiveCyclesMetric::telemetry_path() const {
    return build_fabric_endpoint_path(tray_id_, asic_location_, channel_, "rxActiveCycles");
}

void FabricRxActiveCyclesMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    const auto& samples = telemetry_reader_->get_fabric_telemetry(start_of_update_cycle);
    const auto* sample = find_sample_for_channel(samples, channel_);
    if (sample && sample->snapshot.dynamic_info.has_value()) {
        uint64_t new_value = sample->snapshot.dynamic_info->rx_bandwidth.elapsed_active_cycles;
        changed_since_transmission_ = new_value != value_;
        value_ = new_value;
    }
    set_timestamp_now();
}

std::unordered_map<std::string, std::string> FabricRxActiveCyclesMetric::labels() const {
    return build_ethernet_labels(tray_id_, asic_location_, channel_, link_info_);
}

/**************************************************************************************************
 FabricTxHeartbeatMetric
**************************************************************************************************/

FabricTxHeartbeatMetric::FabricTxHeartbeatMetric(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    uint32_t channel,
    size_t erisc_core,
    std::shared_ptr<FabricTelemetryReader> telemetry_reader,
    const std::unique_ptr<TopologyHelper>& topology_helper) :
    UIntMetric(),
    tray_id_(tray_id),
    asic_location_(asic_location),
    channel_(channel),
    erisc_core_(erisc_core),
    telemetry_reader_(telemetry_reader) {
    value_ = 0;
    link_info_ = get_physical_link_info_for_endpoint(tray_id, asic_location, channel, topology_helper);
}

const std::vector<std::string> FabricTxHeartbeatMetric::telemetry_path() const {
    return build_fabric_erisc_path(tray_id_, asic_location_, channel_, erisc_core_, "txHeartbeat");
}

void FabricTxHeartbeatMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    const auto& samples = telemetry_reader_->get_fabric_telemetry(start_of_update_cycle);
    const auto* sample = find_sample_for_channel(samples, channel_);
    if (sample && sample->snapshot.dynamic_info.has_value()) {
        if (erisc_core_ < sample->snapshot.dynamic_info->erisc.size()) {
            uint64_t new_value = sample->snapshot.dynamic_info->erisc[erisc_core_].tx_heartbeat;
            changed_since_transmission_ = new_value != value_;
            value_ = new_value;
        }
    }
    set_timestamp_now();
}

std::unordered_map<std::string, std::string> FabricTxHeartbeatMetric::labels() const {
    auto labels = build_ethernet_labels(tray_id_, asic_location_, channel_, link_info_);
    labels["erisc_core"] = std::to_string(erisc_core_);
    return labels;
}

/**************************************************************************************************
 FabricRxHeartbeatMetric
**************************************************************************************************/

FabricRxHeartbeatMetric::FabricRxHeartbeatMetric(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    uint32_t channel,
    size_t erisc_core,
    std::shared_ptr<FabricTelemetryReader> telemetry_reader,
    const std::unique_ptr<TopologyHelper>& topology_helper) :
    UIntMetric(),
    tray_id_(tray_id),
    asic_location_(asic_location),
    channel_(channel),
    erisc_core_(erisc_core),
    telemetry_reader_(telemetry_reader) {
    value_ = 0;
    link_info_ = get_physical_link_info_for_endpoint(tray_id, asic_location, channel, topology_helper);
}

const std::vector<std::string> FabricRxHeartbeatMetric::telemetry_path() const {
    return build_fabric_erisc_path(tray_id_, asic_location_, channel_, erisc_core_, "rxHeartbeat");
}

void FabricRxHeartbeatMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    const auto& samples = telemetry_reader_->get_fabric_telemetry(start_of_update_cycle);
    const auto* sample = find_sample_for_channel(samples, channel_);
    if (sample && sample->snapshot.dynamic_info.has_value()) {
        if (erisc_core_ < sample->snapshot.dynamic_info->erisc.size()) {
            uint64_t new_value = sample->snapshot.dynamic_info->erisc[erisc_core_].rx_heartbeat;
            changed_since_transmission_ = new_value != value_;
            value_ = new_value;
        }
    }
    set_timestamp_now();
}

std::unordered_map<std::string, std::string> FabricRxHeartbeatMetric::labels() const {
    auto labels = build_ethernet_labels(tray_id_, asic_location_, channel_, link_info_);
    labels["erisc_core"] = std::to_string(erisc_core_);
    return labels;
}

/**************************************************************************************************
 FabricRouterStateMetric
**************************************************************************************************/

FabricRouterStateMetric::FabricRouterStateMetric(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    uint32_t channel,
    size_t erisc_core,
    std::shared_ptr<FabricTelemetryReader> telemetry_reader,
    const std::unique_ptr<TopologyHelper>& topology_helper) :
    UIntMetric(),
    tray_id_(tray_id),
    asic_location_(asic_location),
    channel_(channel),
    erisc_core_(erisc_core),
    telemetry_reader_(telemetry_reader) {
    value_ = 0;
    link_info_ = get_physical_link_info_for_endpoint(tray_id, asic_location, channel, topology_helper);
}

const std::vector<std::string> FabricRouterStateMetric::telemetry_path() const {
    return build_fabric_erisc_path(tray_id_, asic_location_, channel_, erisc_core_, "routerState");
}

void FabricRouterStateMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    const auto& samples = telemetry_reader_->get_fabric_telemetry(start_of_update_cycle);
    const auto* sample = find_sample_for_channel(samples, channel_);
    if (sample && sample->snapshot.dynamic_info.has_value()) {
        if (erisc_core_ < sample->snapshot.dynamic_info->erisc.size()) {
            uint64_t new_value = static_cast<uint64_t>(sample->snapshot.dynamic_info->erisc[erisc_core_].router_state);
            changed_since_transmission_ = new_value != value_;
            value_ = new_value;
        }
    }
    set_timestamp_now();
}

std::unordered_map<std::string, std::string> FabricRouterStateMetric::labels() const {
    auto labels = build_ethernet_labels(tray_id_, asic_location_, channel_, link_info_);
    labels["erisc_core"] = std::to_string(erisc_core_);
    return labels;
}
