// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <topology/topology.hpp>
#include <algorithm>
#include <optional>
#include <string>
#include <vector>

namespace {

// Constants for Ethernet metric path validation (copied from telemetry_collector.cpp)
static constexpr size_t TRAY_PREFIX_LEN = 4;     // "tray".length()
static constexpr size_t CHIP_PREFIX_LEN = 4;     // "chip".length()
static constexpr size_t CHANNEL_PREFIX_LEN = 7;  // "channel".length()

// Helper functions copied from telemetry_collector.cpp for testing
static bool is_ethernet_metric_path(const std::vector<std::string>& path) {
    if (path.size() < 4) {
        return false;
    }

    // Helper to validate prefix and ensure entire suffix is numeric
    auto is_valid_component = [](std::string_view component, std::string_view prefix) {
        if (component.length() <= prefix.length() || component.rfind(prefix, 0) != 0) {
            return false;
        }
        // Validate all characters after prefix are digits
        return std::all_of(
            component.begin() + prefix.length(), component.end(), [](unsigned char c) { return std::isdigit(c); });
    };

    return is_valid_component(path[0], "tray") && is_valid_component(path[1], "chip") &&
           is_valid_component(path[2], "channel");
}

static std::optional<EthernetEndpoint> parse_ethernet_endpoint(const std::vector<std::string>& path) {
    // First validate the path format using the same logic as is_ethernet_metric_path
    if (!is_ethernet_metric_path(path)) {
        return std::nullopt;
    }

    try {
        // After validation, we know the format is correct and all chars after prefix are digits
        uint32_t tray_id = std::stoul(path[0].substr(TRAY_PREFIX_LEN));
        uint32_t asic_location = std::stoul(path[1].substr(CHIP_PREFIX_LEN));
        uint32_t channel = std::stoul(path[2].substr(CHANNEL_PREFIX_LEN));

        return EthernetEndpoint{tt::tt_metal::TrayID(tray_id), tt::tt_metal::ASICLocation(asic_location), channel};
    } catch (const std::exception&) {
        // Can still fail on overflow
        return std::nullopt;
    }
}

// Test fixture for Ethernet path validation
class EthernetPathValidationTest : public ::testing::Test {};

TEST_F(EthernetPathValidationTest, IsEthernetMetricPath_ValidPath_ReturnsTrue) {
    std::vector<std::string> path = {"tray0", "chip1", "channel2", "link_up"};
    EXPECT_TRUE(is_ethernet_metric_path(path));
}

TEST_F(EthernetPathValidationTest, IsEthernetMetricPath_ValidPathMultiDigit_ReturnsTrue) {
    std::vector<std::string> path = {"tray123", "chip456", "channel789", "metric"};
    EXPECT_TRUE(is_ethernet_metric_path(path));
}

TEST_F(EthernetPathValidationTest, IsEthernetMetricPath_TooShort_ReturnsFalse) {
    std::vector<std::string> path = {"tray0", "chip1", "channel2"};  // Missing metric name
    EXPECT_FALSE(is_ethernet_metric_path(path));
}

TEST_F(EthernetPathValidationTest, IsEthernetMetricPath_MissingPrefix_ReturnsFalse) {
    std::vector<std::string> path = {"0", "chip1", "channel2", "link_up"};
    EXPECT_FALSE(is_ethernet_metric_path(path));
}

TEST_F(EthernetPathValidationTest, IsEthernetMetricPath_WrongPrefix_ReturnsFalse) {
    std::vector<std::string> path = {"tray0", "device1", "channel2", "link_up"};
    EXPECT_FALSE(is_ethernet_metric_path(path));
}

TEST_F(EthernetPathValidationTest, IsEthernetMetricPath_NonNumericSuffix_ReturnsFalse) {
    std::vector<std::string> path = {"tray0", "chip1", "channel2x", "link_up"};
    EXPECT_FALSE(is_ethernet_metric_path(path));
}

TEST_F(EthernetPathValidationTest, IsEthernetMetricPath_NoSuffix_ReturnsFalse) {
    std::vector<std::string> path = {"tray", "chip", "channel", "link_up"};
    EXPECT_FALSE(is_ethernet_metric_path(path));
}

TEST_F(EthernetPathValidationTest, IsEthernetMetricPath_PartialNumericSuffix_ReturnsFalse) {
    std::vector<std::string> path = {"tray5abc", "chip1", "channel2", "link_up"};
    EXPECT_FALSE(is_ethernet_metric_path(path));
}

TEST_F(EthernetPathValidationTest, IsEthernetMetricPath_EmptyPath_ReturnsFalse) {
    std::vector<std::string> path = {};
    EXPECT_FALSE(is_ethernet_metric_path(path));
}

// Test fixture for Ethernet endpoint parsing
class EthernetEndpointParsingTest : public ::testing::Test {};

TEST_F(EthernetEndpointParsingTest, ParseEndpoint_ValidPath_ReturnsEndpoint) {
    std::vector<std::string> path = {"tray5", "chip10", "channel15", "link_up"};

    auto endpoint_opt = parse_ethernet_endpoint(path);

    ASSERT_TRUE(endpoint_opt.has_value());
    EXPECT_EQ(*endpoint_opt.value().tray_id, 5u);
    EXPECT_EQ(*endpoint_opt.value().asic_location, 10u);
    EXPECT_EQ(endpoint_opt.value().channel, 15u);
}

TEST_F(EthernetEndpointParsingTest, ParseEndpoint_TooShort_ReturnsNullopt) {
    std::vector<std::string> path = {"tray0", "chip1", "channel2"};
    EXPECT_FALSE(parse_ethernet_endpoint(path).has_value());
}

TEST_F(EthernetEndpointParsingTest, ParseEndpoint_ComponentTooShort_ReturnsNullopt) {
    std::vector<std::string> path = {"tray", "chip1", "channel2", "metric"};
    EXPECT_FALSE(parse_ethernet_endpoint(path).has_value());
}

TEST_F(EthernetEndpointParsingTest, ParseEndpoint_InvalidNumber_ReturnsNullopt) {
    std::vector<std::string> path = {"trayXYZ", "chip1", "channel2", "metric"};
    EXPECT_FALSE(parse_ethernet_endpoint(path).has_value());
}

TEST_F(EthernetEndpointParsingTest, ParseEndpoint_NegativeNumber_ReturnsNullopt) {
    std::vector<std::string> path = {"tray-5", "chip1", "channel2", "metric"};
    EXPECT_FALSE(parse_ethernet_endpoint(path).has_value());
}

TEST_F(EthernetEndpointParsingTest, ParseEndpoint_Overflow_ReturnsNullopt) {
    // Number too large for uint32_t
    std::vector<std::string> path = {"tray99999999999999999999", "chip1", "channel2", "metric"};
    EXPECT_FALSE(parse_ethernet_endpoint(path).has_value());
}

TEST_F(EthernetEndpointParsingTest, ParseEndpoint_EmptyPath_ReturnsNullopt) {
    std::vector<std::string> path = {};
    EXPECT_FALSE(parse_ethernet_endpoint(path).has_value());
}

TEST_F(EthernetEndpointParsingTest, ParseEndpoint_TrailingCharacters_ReturnsNullopt) {
    std::vector<std::string> path = {"tray5x", "chip1", "channel2", "metric"};
    EXPECT_FALSE(parse_ethernet_endpoint(path).has_value());
}

// Integration test: validate and parse together
TEST_F(EthernetEndpointParsingTest, ValidateThenParse_ValidPath_Succeeds) {
    std::vector<std::string> path = {"tray5", "chip10", "channel15", "link_up"};

    ASSERT_TRUE(is_ethernet_metric_path(path));
    auto endpoint_opt = parse_ethernet_endpoint(path);
    ASSERT_TRUE(endpoint_opt.has_value());
    EXPECT_EQ(*endpoint_opt.value().tray_id, 5u);
    EXPECT_EQ(*endpoint_opt.value().asic_location, 10u);
    EXPECT_EQ(endpoint_opt.value().channel, 15u);
}

TEST_F(EthernetEndpointParsingTest, ValidateThenParse_InvalidPath_Fails) {
    std::vector<std::string> path = {"tray5x", "chip10", "channel15", "link_up"};

    EXPECT_FALSE(is_ethernet_metric_path(path));
    EXPECT_FALSE(parse_ethernet_endpoint(path).has_value());
}

}  // namespace
