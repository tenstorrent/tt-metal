// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <topology/topology.hpp>
#include <telemetry/telemetry_snapshot.hpp>
#include <unordered_set>

namespace {

// Test fixture for EthernetEndpoint tests
class EthernetEndpointTest : public ::testing::Test {
protected:
    EthernetEndpoint endpoint1{tt::tt_metal::TrayID(0), tt::tt_metal::ASICLocation(1), 2};
    EthernetEndpoint endpoint2{tt::tt_metal::TrayID(0), tt::tt_metal::ASICLocation(1), 2};
    EthernetEndpoint endpoint3{tt::tt_metal::TrayID(1), tt::tt_metal::ASICLocation(1), 2};
};

TEST_F(EthernetEndpointTest, EqualityOperator_SameValues_ReturnsTrue) { EXPECT_EQ(endpoint1, endpoint2); }

TEST_F(EthernetEndpointTest, EqualityOperator_DifferentTray_ReturnsFalse) { EXPECT_NE(endpoint1, endpoint3); }

TEST_F(EthernetEndpointTest, Hash_SameValues_ProducesSameHash) {
    EthernetEndpointHash hasher;
    EXPECT_EQ(hasher(endpoint1), hasher(endpoint2));
}

TEST_F(EthernetEndpointTest, Hash_DifferentValues_ProducesDifferentHash) {
    EthernetEndpointHash hasher;
    EXPECT_NE(hasher(endpoint1), hasher(endpoint3));
}

TEST_F(EthernetEndpointTest, Hash_CanBeUsedInUnorderedSet) {
    std::unordered_set<EthernetEndpoint, EthernetEndpointHash> endpoint_set;
    endpoint_set.insert(endpoint1);
    endpoint_set.insert(endpoint2);  // Same as endpoint1
    endpoint_set.insert(endpoint3);  // Different

    EXPECT_EQ(endpoint_set.size(), 2);  // Only 2 unique endpoints
    EXPECT_TRUE(endpoint_set.find(endpoint1) != endpoint_set.end());
    EXPECT_TRUE(endpoint_set.find(endpoint3) != endpoint_set.end());
}

// Test fixture for RemoteEndpointInfo tests
class RemoteEndpointInfoTest : public ::testing::Test {
protected:
    RemoteEndpointInfo remote1{
        .hostname = "host1",
        .tray = tt::tt_metal::TrayID(0),
        .asic = tt::tt_metal::ASICLocation(1),
        .channel = 2,
        .aisle = "A1",
        .rack = 10};
    RemoteEndpointInfo remote2{
        .hostname = "host1",
        .tray = tt::tt_metal::TrayID(0),
        .asic = tt::tt_metal::ASICLocation(1),
        .channel = 2,
        .aisle = "A1",
        .rack = 10};
    RemoteEndpointInfo remote3{
        .hostname = "host2",
        .tray = tt::tt_metal::TrayID(0),
        .asic = tt::tt_metal::ASICLocation(1),
        .channel = 2,
        .aisle = "A2",
        .rack = 11};
};

TEST_F(RemoteEndpointInfoTest, EqualityOperator_SameValues_ReturnsTrue) { EXPECT_EQ(remote1, remote2); }

TEST_F(RemoteEndpointInfoTest, EqualityOperator_DifferentValues_ReturnsFalse) { EXPECT_NE(remote1, remote3); }

// Test fixture for PhysicalLinkInfo tests
class PhysicalLinkInfoTest : public ::testing::Test {
protected:
    tt::scaleout_tools::PortType port_type = tt::scaleout_tools::PortType::QSFP_DD;
    tt::scaleout_tools::PortId port_id{5};
    RemoteEndpointInfo remote_info{
        .hostname = "remote-host",
        .tray = tt::tt_metal::TrayID(1),
        .asic = tt::tt_metal::ASICLocation(2),
        .channel = 3,
        .aisle = "B2",
        .rack = 20};
};

TEST_F(PhysicalLinkInfoTest, CreateInternal_SetsCorrectProperties) {
    auto link_info = PhysicalLinkInfo::create(port_type, port_id);

    EXPECT_EQ(link_info.port_type, port_type);
    EXPECT_EQ(*link_info.port_id, *port_id);
    EXPECT_FALSE(link_info.is_external());
    EXPECT_FALSE(link_info.remote_endpoint.has_value());
}

TEST_F(PhysicalLinkInfoTest, CreateExternal_SetsCorrectProperties) {
    auto link_info = PhysicalLinkInfo::create(port_type, port_id, remote_info);

    EXPECT_EQ(link_info.port_type, port_type);
    EXPECT_EQ(*link_info.port_id, *port_id);
    EXPECT_TRUE(link_info.is_external());
    ASSERT_TRUE(link_info.remote_endpoint.has_value());
    EXPECT_EQ(link_info.remote_endpoint.value(), remote_info);
}

TEST_F(PhysicalLinkInfoTest, IsExternal_InternalLink_ReturnsFalse) {
    auto link_info = PhysicalLinkInfo::create(port_type, port_id);
    EXPECT_FALSE(link_info.is_external());
}

TEST_F(PhysicalLinkInfoTest, IsExternal_ExternalLink_ReturnsTrue) {
    auto link_info = PhysicalLinkInfo::create(port_type, port_id, remote_info);
    EXPECT_TRUE(link_info.is_external());
}

TEST_F(PhysicalLinkInfoTest, EqualityOperator_InternalLinks_SameValues_ReturnsTrue) {
    auto link1 = PhysicalLinkInfo::create(port_type, port_id);
    auto link2 = PhysicalLinkInfo::create(port_type, port_id);

    EXPECT_EQ(link1, link2);
}

TEST_F(PhysicalLinkInfoTest, EqualityOperator_ExternalLinks_SameValues_ReturnsTrue) {
    auto link1 = PhysicalLinkInfo::create(port_type, port_id, remote_info);
    auto link2 = PhysicalLinkInfo::create(port_type, port_id, remote_info);

    EXPECT_EQ(link1, link2);
}

TEST_F(PhysicalLinkInfoTest, EqualityOperator_InternalVsExternal_ReturnsFalse) {
    auto internal_link = PhysicalLinkInfo::create(port_type, port_id);
    auto external_link = PhysicalLinkInfo::create(port_type, port_id, remote_info);

    EXPECT_NE(internal_link, external_link);
}

TEST_F(PhysicalLinkInfoTest, Invariant_InternalHasNoRemote) {
    auto link_info = PhysicalLinkInfo::create(port_type, port_id);

    // Invariant: internal links should never have remote_endpoint
    EXPECT_FALSE(link_info.is_external());
    EXPECT_FALSE(link_info.remote_endpoint.has_value());
}

TEST_F(PhysicalLinkInfoTest, Invariant_ExternalHasRemote) {
    auto link_info = PhysicalLinkInfo::create(port_type, port_id, remote_info);

    // Invariant: external links should always have remote_endpoint
    EXPECT_TRUE(link_info.is_external());
    EXPECT_TRUE(link_info.remote_endpoint.has_value());
}

// Test fixture for TelemetrySnapshot physical_link_info merging
class TelemetrySnapshotPhysicalLinkInfoTest : public ::testing::Test {
protected:
    nlohmann::json create_link_info_json(bool is_external) {
        nlohmann::json j;
        j["port_type"] = 1;
        j["port_id"] = 5;
        j["is_external"] = is_external;
        if (is_external) {
            j["remote_hostname"] = "remote-host";
            j["remote_tray"] = 1;
            j["remote_asic"] = 2;
            j["remote_channel"] = 3;
            j["remote_aisle"] = "A1";
            j["remote_rack"] = 10;
        }
        return j;
    }
};

TEST_F(TelemetrySnapshotPhysicalLinkInfoTest, MergeFast_AddsNewPhysicalLinkInfo) {
    TelemetrySnapshot snapshot1;
    TelemetrySnapshot snapshot2;

    snapshot2.physical_link_info["host1/tray0/chip1/channel2/link_up"] = create_link_info_json(true);

    snapshot1.merge_from(snapshot2, false);

    EXPECT_EQ(snapshot1.physical_link_info.size(), 1);
    EXPECT_TRUE(snapshot1.physical_link_info.contains("host1/tray0/chip1/channel2/link_up"));
}

TEST_F(TelemetrySnapshotPhysicalLinkInfoTest, MergeFast_PreservesExistingPhysicalLinkInfo) {
    TelemetrySnapshot snapshot1;
    TelemetrySnapshot snapshot2;

    auto existing_info = create_link_info_json(false);
    auto new_info = create_link_info_json(true);

    snapshot1.physical_link_info["host1/tray0/chip1/channel2/link_up"] = existing_info;
    snapshot2.physical_link_info["host1/tray0/chip1/channel2/link_up"] = new_info;

    snapshot1.merge_from(snapshot2, false);

    // Fast merge uses insert, which doesn't overwrite existing keys
    EXPECT_EQ(snapshot1.physical_link_info["host1/tray0/chip1/channel2/link_up"], existing_info);
}

TEST_F(TelemetrySnapshotPhysicalLinkInfoTest, MergeWithValidation_DetectsDifferentValues) {
    TelemetrySnapshot snapshot1;
    TelemetrySnapshot snapshot2;

    auto info1 = create_link_info_json(false);
    auto info2 = create_link_info_json(true);  // Different value

    snapshot1.physical_link_info["host1/tray0/chip1/channel2/link_up"] = info1;
    snapshot2.physical_link_info["host1/tray0/chip1/channel2/link_up"] = info2;

    // This should log an error but not crash
    snapshot1.merge_from(snapshot2, true);

    // Validation path also doesn't overwrite
    EXPECT_EQ(snapshot1.physical_link_info["host1/tray0/chip1/channel2/link_up"], info1);
}

TEST_F(TelemetrySnapshotPhysicalLinkInfoTest, MergeWithValidation_AllowsSameValues) {
    TelemetrySnapshot snapshot1;
    TelemetrySnapshot snapshot2;

    auto info = create_link_info_json(true);

    snapshot1.physical_link_info["host1/tray0/chip1/channel2/link_up"] = info;
    snapshot2.physical_link_info["host1/tray0/chip1/channel2/link_up"] = info;

    // Should succeed without error since values match
    snapshot1.merge_from(snapshot2, true);

    EXPECT_EQ(snapshot1.physical_link_info["host1/tray0/chip1/channel2/link_up"], info);
}

TEST_F(TelemetrySnapshotPhysicalLinkInfoTest, Clear_RemovesPhysicalLinkInfo) {
    TelemetrySnapshot snapshot;
    snapshot.physical_link_info["host1/tray0/chip1/channel2/link_up"] = create_link_info_json(true);

    EXPECT_EQ(snapshot.physical_link_info.size(), 1);

    snapshot.clear();

    EXPECT_EQ(snapshot.physical_link_info.size(), 0);
}

TEST_F(TelemetrySnapshotPhysicalLinkInfoTest, ToJson_IncludesPhysicalLinkInfo) {
    TelemetrySnapshot snapshot;
    snapshot.physical_link_info["host1/tray0/chip1/channel2/link_up"] = create_link_info_json(true);

    nlohmann::json j;
    to_json(j, snapshot);

    EXPECT_TRUE(j.contains("physical_link_info"));
    EXPECT_EQ(j["physical_link_info"].size(), 1);
}

TEST_F(TelemetrySnapshotPhysicalLinkInfoTest, FromJson_WithPhysicalLinkInfo_Deserializes) {
    nlohmann::json j;
    j["bool_metrics"] = nlohmann::json::object();
    j["uint_metrics"] = nlohmann::json::object();
    j["double_metrics"] = nlohmann::json::object();
    j["string_metrics"] = nlohmann::json::object();
    j["bool_metric_timestamps"] = nlohmann::json::object();
    j["uint_metric_units"] = nlohmann::json::object();
    j["uint_metric_timestamps"] = nlohmann::json::object();
    j["double_metric_units"] = nlohmann::json::object();
    j["double_metric_timestamps"] = nlohmann::json::object();
    j["string_metric_units"] = nlohmann::json::object();
    j["string_metric_timestamps"] = nlohmann::json::object();
    j["metric_labels"] = nlohmann::json::object();
    // Maps with integer keys are serialized as arrays in nlohmann::json
    j["metric_unit_display_label_by_code"] = nlohmann::json::array();
    j["metric_unit_full_label_by_code"] = nlohmann::json::array();
    j["physical_link_info"] = nlohmann::json::object();
    j["physical_link_info"]["host1/tray0/chip1/channel2/link_up"] = create_link_info_json(true);

    TelemetrySnapshot snapshot;
    from_json(j, snapshot);

    EXPECT_EQ(snapshot.physical_link_info.size(), 1);
    EXPECT_TRUE(snapshot.physical_link_info.contains("host1/tray0/chip1/channel2/link_up"));
}

TEST_F(TelemetrySnapshotPhysicalLinkInfoTest, FromJson_WithoutPhysicalLinkInfo_HandlesGracefully) {
    // Simulate older snapshot format without physical_link_info field
    nlohmann::json j;
    j["bool_metrics"] = nlohmann::json::object();
    j["uint_metrics"] = nlohmann::json::object();
    j["double_metrics"] = nlohmann::json::object();
    j["string_metrics"] = nlohmann::json::object();
    j["bool_metric_timestamps"] = nlohmann::json::object();
    j["uint_metric_units"] = nlohmann::json::object();
    j["uint_metric_timestamps"] = nlohmann::json::object();
    j["double_metric_units"] = nlohmann::json::object();
    j["double_metric_timestamps"] = nlohmann::json::object();
    j["string_metric_units"] = nlohmann::json::object();
    j["string_metric_timestamps"] = nlohmann::json::object();
    j["metric_labels"] = nlohmann::json::object();
    // Maps with integer keys are serialized as arrays in nlohmann::json
    j["metric_unit_display_label_by_code"] = nlohmann::json::array();
    j["metric_unit_full_label_by_code"] = nlohmann::json::array();
    // Note: physical_link_info is NOT present

    TelemetrySnapshot snapshot;
    from_json(j, snapshot);

    // Should deserialize without error
    EXPECT_EQ(snapshot.physical_link_info.size(), 0);
}

// Test fixture for ASICLocationAndTrayIDHash
class ASICLocationAndTrayIDHashTest : public ::testing::Test {
protected:
    std::pair<tt::tt_metal::ASICLocation, tt::tt_metal::TrayID> pair1{
        tt::tt_metal::ASICLocation(1), tt::tt_metal::TrayID(2)};
    std::pair<tt::tt_metal::ASICLocation, tt::tt_metal::TrayID> pair2{
        tt::tt_metal::ASICLocation(1), tt::tt_metal::TrayID(2)};
    std::pair<tt::tt_metal::ASICLocation, tt::tt_metal::TrayID> pair3{
        tt::tt_metal::ASICLocation(3), tt::tt_metal::TrayID(4)};
};

TEST_F(ASICLocationAndTrayIDHashTest, Hash_SameValues_ProducesSameHash) {
    ASICLocationAndTrayIDHash hasher;
    EXPECT_EQ(hasher(pair1), hasher(pair2));
}

TEST_F(ASICLocationAndTrayIDHashTest, Hash_DifferentValues_ProducesDifferentHash) {
    ASICLocationAndTrayIDHash hasher;
    EXPECT_NE(hasher(pair1), hasher(pair3));
}

TEST_F(ASICLocationAndTrayIDHashTest, Hash_CanBeUsedInUnorderedMap) {
    std::unordered_map<std::pair<tt::tt_metal::ASICLocation, tt::tt_metal::TrayID>, int, ASICLocationAndTrayIDHash>
        test_map;

    test_map[pair1] = 100;
    test_map[pair2] = 200;  // Same key, should overwrite
    test_map[pair3] = 300;

    EXPECT_EQ(test_map.size(), 2);  // Only 2 unique keys
    EXPECT_EQ(test_map[pair1], 200);
    EXPECT_EQ(test_map[pair3], 300);
}

}  // namespace
