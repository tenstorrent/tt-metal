// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <topology/topology.hpp>
#include <telemetry/metric.hpp>
#include <nlohmann/json.hpp>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// Helper: Create PhysicalLinkInfo for testing
static PhysicalLinkInfo create_test_internal_link() {
    return PhysicalLinkInfo::create(tt::scaleout_tools::PortType::QSFP_DD, tt::scaleout_tools::PortId{5});
}

static PhysicalLinkInfo create_test_external_link() {
    RemoteEndpointInfo remote{
        .hostname = "remote-host",
        .tray = tt::tt_metal::TrayID(1),
        .asic = tt::tt_metal::ASICLocation(2),
        .channel = 3,
        .aisle = "A1",
        .rack = 10};
    return PhysicalLinkInfo::create(
        tt::scaleout_tools::PortType::QSFP_DD, tt::scaleout_tools::PortId{5}, std::move(remote));
}

// Copy of physical_link_info_to_json from telemetry_collector.cpp for testing
static nlohmann::json physical_link_info_to_json(const PhysicalLinkInfo& link_info) {
    nlohmann::json link_json;
    link_json["port_type"] = static_cast<int>(link_info.port_type);
    link_json["port_id"] = *link_info.port_id;
    link_json["is_external"] = link_info.is_external();

    if (link_info.remote_endpoint.has_value()) {
        const auto& remote = link_info.remote_endpoint.value();
        link_json["remote_hostname"] = remote.hostname;
        link_json["remote_tray"] = *remote.tray;
        link_json["remote_asic"] = *remote.asic;
        link_json["remote_channel"] = remote.channel;
        link_json["remote_aisle"] = remote.aisle;
        link_json["remote_rack"] = remote.rack;
    }

    return link_json;
}

// Test fixture for physical_link_info_to_json
class PhysicalLinkInfoToJsonTest : public ::testing::Test {};

TEST_F(PhysicalLinkInfoToJsonTest, InternalLink_ProducesCorrectJson) {
    auto link_info = create_test_internal_link();

    nlohmann::json result = physical_link_info_to_json(link_info);

    EXPECT_EQ(result["port_type"], static_cast<int>(tt::scaleout_tools::PortType::QSFP_DD));
    EXPECT_EQ(result["port_id"], 5);
    EXPECT_EQ(result["is_external"], false);
    // Internal links should not have remote fields
    EXPECT_FALSE(result.contains("remote_hostname"));
    EXPECT_FALSE(result.contains("remote_tray"));
    EXPECT_FALSE(result.contains("remote_asic"));
    EXPECT_FALSE(result.contains("remote_channel"));
    EXPECT_FALSE(result.contains("remote_aisle"));
    EXPECT_FALSE(result.contains("remote_rack"));
}

TEST_F(PhysicalLinkInfoToJsonTest, ExternalLink_ProducesCorrectJson) {
    auto link_info = create_test_external_link();

    nlohmann::json result = physical_link_info_to_json(link_info);

    EXPECT_EQ(result["port_type"], static_cast<int>(tt::scaleout_tools::PortType::QSFP_DD));
    EXPECT_EQ(result["port_id"], 5);
    EXPECT_EQ(result["is_external"], true);
    // External links should have all remote fields
    EXPECT_EQ(result["remote_hostname"], "remote-host");
    EXPECT_EQ(result["remote_tray"], 1);
    EXPECT_EQ(result["remote_asic"], 2);
    EXPECT_EQ(result["remote_channel"], 3);
    EXPECT_EQ(result["remote_aisle"], "A1");
    EXPECT_EQ(result["remote_rack"], 10);
}

TEST_F(PhysicalLinkInfoToJsonTest, IsExternalField_UsesMethodCall) {
    auto internal_link = create_test_internal_link();
    auto external_link = create_test_external_link();

    nlohmann::json internal_json = physical_link_info_to_json(internal_link);
    nlohmann::json external_json = physical_link_info_to_json(external_link);

    // Verify is_external field matches the is_external() method result
    EXPECT_EQ(internal_json["is_external"], internal_link.is_external());
    EXPECT_EQ(external_json["is_external"], external_link.is_external());
}

TEST_F(PhysicalLinkInfoToJsonTest, JsonCanRoundTrip) {
    auto link_info = create_test_external_link();

    nlohmann::json json_output = physical_link_info_to_json(link_info);
    std::string serialized = json_output.dump();
    nlohmann::json deserialized = nlohmann::json::parse(serialized);

    // Verify all fields preserved
    EXPECT_EQ(deserialized["port_type"], static_cast<int>(tt::scaleout_tools::PortType::QSFP_DD));
    EXPECT_EQ(deserialized["port_id"], 5);
    EXPECT_EQ(deserialized["is_external"], true);
    EXPECT_EQ(deserialized["remote_hostname"], "remote-host");
}

// Mock metric for testing
class MockMetric : public UIntMetric {
private:
    std::vector<std::string> path_;

public:
    explicit MockMetric(std::vector<std::string> path) : UIntMetric(), path_(std::move(path)) {}

    const std::vector<std::string> telemetry_path() const override { return path_; }

    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override {
        // No-op for testing
    }
};

// Mock TopologyHelper for testing populate functions
class MockTopologyHelper {
public:
    std::unordered_map<EthernetEndpoint, PhysicalLinkInfo, EthernetEndpointHash> endpoint_map;

    std::optional<PhysicalLinkInfo> get_physical_link_info(const EthernetEndpoint& endpoint) const {
        auto it = endpoint_map.find(endpoint);
        if (it != endpoint_map.end()) {
            return it->second;
        }
        return std::nullopt;
    }
};

// Constants for path parsing (copied from telemetry_collector.cpp)
static constexpr size_t TRAY_PREFIX_LEN = 4;
static constexpr size_t CHIP_PREFIX_LEN = 4;
static constexpr size_t CHANNEL_PREFIX_LEN = 7;

// Helper functions copied from telemetry_collector.cpp
static bool is_ethernet_metric_path(const std::vector<std::string>& path) {
    if (path.size() < 4) {
        return false;
    }

    auto is_valid_component = [](std::string_view component, std::string_view prefix) {
        if (component.length() <= prefix.length() || component.rfind(prefix, 0) != 0) {
            return false;
        }
        return std::all_of(
            component.begin() + prefix.length(), component.end(), [](unsigned char c) { return std::isdigit(c); });
    };

    return is_valid_component(path[0], "tray") && is_valid_component(path[1], "chip") &&
           is_valid_component(path[2], "channel");
}

static std::optional<EthernetEndpoint> parse_ethernet_endpoint(const std::vector<std::string>& path) {
    if (path.size() < 4 || path[0].length() <= TRAY_PREFIX_LEN || path[1].length() <= CHIP_PREFIX_LEN ||
        path[2].length() <= CHANNEL_PREFIX_LEN) {
        return std::nullopt;
    }

    try {
        uint32_t tray_id = std::stoul(path[0].substr(TRAY_PREFIX_LEN));
        uint32_t asic_location = std::stoul(path[1].substr(CHIP_PREFIX_LEN));
        uint32_t channel = std::stoul(path[2].substr(CHANNEL_PREFIX_LEN));

        return EthernetEndpoint{tt::tt_metal::TrayID(tray_id), tt::tt_metal::ASICLocation(asic_location), channel};
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

static void populate_physical_link_info_for_metric(
    std::string_view metric_path,
    const std::vector<std::string>& telemetry_path,
    const MockTopologyHelper* topology_translation,
    std::unordered_map<std::string, nlohmann::json>& physical_link_info_map) {
    if (!topology_translation) {
        return;
    }

    if (!is_ethernet_metric_path(telemetry_path)) {
        return;
    }

    auto endpoint_opt = parse_ethernet_endpoint(telemetry_path);
    if (!endpoint_opt.has_value()) {
        return;
    }

    auto link_info_opt = topology_translation->get_physical_link_info(endpoint_opt.value());
    if (link_info_opt.has_value()) {
        physical_link_info_map[std::string(metric_path)] = physical_link_info_to_json(link_info_opt.value());
    }
}

// Test fixture for populate_physical_link_info_for_metric
class PopulatePhysicalLinkInfoTest : public ::testing::Test {
protected:
    MockTopologyHelper mock_topology;
    std::unordered_map<std::string, nlohmann::json> link_info_map;

    void SetUp() override {
        // Pre-populate mock topology with test data
        EthernetEndpoint endpoint{tt::tt_metal::TrayID(0), tt::tt_metal::ASICLocation(1), 2};
        mock_topology.endpoint_map[endpoint] = create_test_external_link();
    }
};

TEST_F(PopulatePhysicalLinkInfoTest, ValidEthernetMetric_PopulatesLinkInfo) {
    std::vector<std::string> path = {"tray0", "chip1", "channel2", "link_up"};
    std::string metric_path = "host1/tray0/chip1/channel2/link_up";

    populate_physical_link_info_for_metric(metric_path, path, &mock_topology, link_info_map);

    EXPECT_EQ(link_info_map.size(), 1);
    EXPECT_TRUE(link_info_map.contains(metric_path));
    EXPECT_EQ(link_info_map[metric_path]["is_external"], true);
}

TEST_F(PopulatePhysicalLinkInfoTest, NonEthernetMetric_SkipsPopulation) {
    std::vector<std::string> path = {"system", "cpu", "temperature"};
    std::string metric_path = "host1/system/cpu/temperature";

    populate_physical_link_info_for_metric(metric_path, path, &mock_topology, link_info_map);

    EXPECT_EQ(link_info_map.size(), 0);
}

TEST_F(PopulatePhysicalLinkInfoTest, NullTopology_SkipsPopulation) {
    std::vector<std::string> path = {"tray0", "chip1", "channel2", "link_up"};
    std::string metric_path = "host1/tray0/chip1/channel2/link_up";

    populate_physical_link_info_for_metric(metric_path, path, nullptr, link_info_map);

    EXPECT_EQ(link_info_map.size(), 0);
}

TEST_F(PopulatePhysicalLinkInfoTest, UnparseablePath_SkipsPopulation) {
    std::vector<std::string> path = {"trayXYZ", "chip1", "channel2", "link_up"};
    std::string metric_path = "host1/trayXYZ/chip1/channel2/link_up";

    populate_physical_link_info_for_metric(metric_path, path, &mock_topology, link_info_map);

    EXPECT_EQ(link_info_map.size(), 0);
}

TEST_F(PopulatePhysicalLinkInfoTest, EndpointNotFound_SkipsPopulation) {
    std::vector<std::string> path = {"tray99", "chip99", "channel99", "link_up"};  // Not in mock_topology
    std::string metric_path = "host1/tray99/chip99/channel99/link_up";

    populate_physical_link_info_for_metric(metric_path, path, &mock_topology, link_info_map);

    EXPECT_EQ(link_info_map.size(), 0);
}

TEST_F(PopulatePhysicalLinkInfoTest, MultipleMetrics_PopulatesAll) {
    // Add another endpoint to mock topology
    EthernetEndpoint endpoint2{tt::tt_metal::TrayID(1), tt::tt_metal::ASICLocation(2), 3};
    mock_topology.endpoint_map[endpoint2] = create_test_internal_link();

    // Populate first metric
    std::vector<std::string> path1 = {"tray0", "chip1", "channel2", "link_up"};
    populate_physical_link_info_for_metric("host1/tray0/chip1/channel2/link_up", path1, &mock_topology, link_info_map);

    // Populate second metric
    std::vector<std::string> path2 = {"tray1", "chip2", "channel3", "link_up"};
    populate_physical_link_info_for_metric("host1/tray1/chip2/channel3/link_up", path2, &mock_topology, link_info_map);

    EXPECT_EQ(link_info_map.size(), 2);
    EXPECT_TRUE(link_info_map.contains("host1/tray0/chip1/channel2/link_up"));
    EXPECT_TRUE(link_info_map.contains("host1/tray1/chip2/channel3/link_up"));
}

// Helper: Create a metric path string from components
static std::string make_metric_path(const std::vector<std::string>& components) {
    std::string path;
    for (size_t i = 0; i < components.size(); ++i) {
        if (i > 0) {
            path += "/";
        }
        path += components[i];
    }
    return path;
}

// Mock metric implementation for batch testing
static std::string get_cluster_wide_telemetry_path_for_test(
    const std::string& hostname, const std::vector<std::string>& local_path) {
    std::vector<std::string> full_path = {hostname};
    full_path.insert(full_path.end(), local_path.begin(), local_path.end());
    return make_metric_path(full_path);
}

// Test fixture for populate_physical_link_info_for_metrics (batch)
class PopulatePhysicalLinkInfoBatchTest : public ::testing::Test {
protected:
    MockTopologyHelper mock_topology;
    std::unordered_map<std::string, nlohmann::json> link_info_map;
    std::vector<std::unique_ptr<MockMetric>> metrics;

    void SetUp() override {
        // Pre-populate mock topology
        EthernetEndpoint ep1{tt::tt_metal::TrayID(0), tt::tt_metal::ASICLocation(1), 2};
        mock_topology.endpoint_map[ep1] = create_test_external_link();

        // Create test metrics
        metrics.push_back(
            std::make_unique<MockMetric>(std::vector<std::string>{"tray0", "chip1", "channel2", "link_up"}));
        metrics.push_back(
            std::make_unique<MockMetric>(std::vector<std::string>{"system", "cpu", "temp"}));  // Non-Ethernet
        metrics.push_back(
            std::make_unique<MockMetric>(std::vector<std::string>{"tray0", "chip1", "channel2", "bandwidth"}));
    }

    void populate_for_test() {
        for (const auto& metric : metrics) {
            std::string path = get_cluster_wide_telemetry_path_for_test("host1", metric->telemetry_path());
            populate_physical_link_info_for_metric(path, metric->telemetry_path(), &mock_topology, link_info_map);
        }
    }
};

TEST_F(PopulatePhysicalLinkInfoBatchTest, MixedMetrics_OnlyPopulatesEthernetMetrics) {
    populate_for_test();

    // Should populate 2 Ethernet metrics (link_up and bandwidth), skip 1 non-Ethernet
    EXPECT_EQ(link_info_map.size(), 2);
    EXPECT_TRUE(link_info_map.contains("host1/tray0/chip1/channel2/link_up"));
    EXPECT_TRUE(link_info_map.contains("host1/tray0/chip1/channel2/bandwidth"));
    EXPECT_FALSE(link_info_map.contains("host1/system/cpu/temp"));
}

TEST_F(PopulatePhysicalLinkInfoBatchTest, EmptyMetricsList_ProducesEmptyMap) {
    metrics.clear();
    populate_for_test();

    EXPECT_EQ(link_info_map.size(), 0);
}

TEST_F(PopulatePhysicalLinkInfoBatchTest, AllNonEthernetMetrics_ProducesEmptyMap) {
    metrics.clear();
    metrics.push_back(std::make_unique<MockMetric>(std::vector<std::string>{"system", "cpu", "temp"}));
    metrics.push_back(std::make_unique<MockMetric>(std::vector<std::string>{"arc", "device0", "voltage"}));

    populate_for_test();

    EXPECT_EQ(link_info_map.size(), 0);
}

TEST_F(PopulatePhysicalLinkInfoBatchTest, SameEndpointDifferentMetrics_PopulatesBoth) {
    // Both metrics target the same endpoint but different metric names
    metrics.clear();
    metrics.push_back(std::make_unique<MockMetric>(std::vector<std::string>{"tray0", "chip1", "channel2", "link_up"}));
    metrics.push_back(
        std::make_unique<MockMetric>(std::vector<std::string>{"tray0", "chip1", "channel2", "bandwidth"}));

    populate_for_test();

    // Both should get the same physical link info
    EXPECT_EQ(link_info_map.size(), 2);
    EXPECT_EQ(
        link_info_map["host1/tray0/chip1/channel2/link_up"], link_info_map["host1/tray0/chip1/channel2/bandwidth"]);
}

// Test fixture for TopologyHelper::get_physical_link_info API
class TopologyHelperApiTest : public ::testing::Test {
protected:
    // Note: We can't easily construct a real TopologyHelper without full UMD/FSD setup,
    // so these tests verify the API contract using the mock topology helper above.
    // The actual TopologyHelper is integration-tested via the FSD parsing logic.
};

TEST_F(TopologyHelperApiTest, MockTopology_LookupExistingEndpoint_ReturnsLinkInfo) {
    MockTopologyHelper topology;
    EthernetEndpoint endpoint{tt::tt_metal::TrayID(0), tt::tt_metal::ASICLocation(1), 2};
    topology.endpoint_map[endpoint] = create_test_external_link();

    auto result = topology.get_physical_link_info(endpoint);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value().is_external());
    EXPECT_EQ(*result.value().port_id, 5);
}

TEST_F(TopologyHelperApiTest, MockTopology_LookupNonExistentEndpoint_ReturnsNullopt) {
    MockTopologyHelper topology;
    EthernetEndpoint endpoint{tt::tt_metal::TrayID(99), tt::tt_metal::ASICLocation(99), 99};

    auto result = topology.get_physical_link_info(endpoint);

    EXPECT_FALSE(result.has_value());
}

TEST_F(TopologyHelperApiTest, MockTopology_MultipleLookups_ReturnsConsistentResults) {
    MockTopologyHelper topology;
    EthernetEndpoint endpoint{tt::tt_metal::TrayID(0), tt::tt_metal::ASICLocation(1), 2};
    topology.endpoint_map[endpoint] = create_test_internal_link();

    auto result1 = topology.get_physical_link_info(endpoint);
    auto result2 = topology.get_physical_link_info(endpoint);

    ASSERT_TRUE(result1.has_value());
    ASSERT_TRUE(result2.has_value());
    EXPECT_EQ(result1.value(), result2.value());
}

TEST_F(TopologyHelperApiTest, MockTopology_ConstCorrectness) {
    MockTopologyHelper topology;
    EthernetEndpoint endpoint{tt::tt_metal::TrayID(0), tt::tt_metal::ASICLocation(1), 2};
    topology.endpoint_map[endpoint] = create_test_external_link();

    // Verify the method is const
    const MockTopologyHelper& const_topology = topology;
    auto result = const_topology.get_physical_link_info(endpoint);

    EXPECT_TRUE(result.has_value());
}

}  // namespace
