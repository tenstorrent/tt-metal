// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <telemetry/metric.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Mock metric that can be configured to throw exceptions during update()
class ThrowingMockMetric : public UIntMetric {
private:
    std::string path_;
    bool should_throw_;
    int update_count_ = 0;

public:
    ThrowingMockMetric(std::string path, bool should_throw) :
        UIntMetric(), path_(std::move(path)), should_throw_(should_throw) {}

    const std::vector<std::string> telemetry_path() const override {
        // Split path by '/' for testing
        std::vector<std::string> components;
        std::string current;
        for (char c : path_) {
            if (c == '/') {
                if (!current.empty()) {
                    components.push_back(current);
                    current.clear();
                }
            } else {
                current += c;
            }
        }
        if (!current.empty()) {
            components.push_back(current);
        }
        return components;
    }

    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override {
        if (should_throw_) {
            throw std::runtime_error("Simulated UMD timeout exception");
        }
        // Normal update: increment value
        value_++;
        update_count_++;
        changed_since_transmission_ = true;
    }

    int get_update_count() const { return update_count_; }
};

// Template helper function copied from telemetry_collector.cpp for testing
template <typename MetricType>
static size_t update_metrics_with_exception_handling(
    std::vector<std::unique_ptr<MetricType>>& metrics,
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    std::chrono::steady_clock::time_point start_of_update_cycle,
    std::string_view metric_type_name) {
    size_t failed_count = 0;

    for (auto& metric : metrics) {
        try {
            metric->update(cluster, start_of_update_cycle);
        } catch (const std::exception& e) {
            failed_count++;
            // In real code, this would log. For tests, we just count.
        }
    }

    return failed_count;
}

// Test fixture
class ExceptionHandlingTest : public ::testing::Test {
protected:
    std::unique_ptr<tt::umd::Cluster> null_cluster_;  // nullptr is fine for these tests
    std::chrono::steady_clock::time_point test_time_ = std::chrono::steady_clock::now();
};

TEST_F(ExceptionHandlingTest, MetricThrowsException_ContinuesExecution) {
    // Create a mix of normal and throwing metrics
    std::vector<std::unique_ptr<ThrowingMockMetric>> metrics;
    metrics.push_back(std::make_unique<ThrowingMockMetric>("tray0/chip0/metric1", false));
    metrics.push_back(std::make_unique<ThrowingMockMetric>("tray0/chip0/metric2", false));
    metrics.push_back(std::make_unique<ThrowingMockMetric>("tray0/chip1/metric3", true));  // throws
    metrics.push_back(std::make_unique<ThrowingMockMetric>("tray0/chip2/metric4", false));
    metrics.push_back(std::make_unique<ThrowingMockMetric>("tray0/chip2/metric5", false));

    // Call the exception handling wrapper
    size_t failed_count = update_metrics_with_exception_handling(metrics, null_cluster_, test_time_, "test");

    // Verify one failure was counted
    EXPECT_EQ(failed_count, 1);

    // Verify non-throwing metrics were updated
    EXPECT_EQ(metrics[0]->get_update_count(), 1);
    EXPECT_EQ(metrics[1]->get_update_count(), 1);
    EXPECT_EQ(metrics[2]->get_update_count(), 0);  // This one threw, so no update
    EXPECT_EQ(metrics[3]->get_update_count(), 1);
    EXPECT_EQ(metrics[4]->get_update_count(), 1);
}

TEST_F(ExceptionHandlingTest, MultipleExceptions_TracksAllFailures) {
    // Create mostly throwing metrics
    std::vector<std::unique_ptr<ThrowingMockMetric>> metrics;
    metrics.push_back(std::make_unique<ThrowingMockMetric>("tray0/chip0/metric1", false));
    metrics.push_back(std::make_unique<ThrowingMockMetric>("tray0/chip1/metric2", true));  // throws
    metrics.push_back(std::make_unique<ThrowingMockMetric>("tray0/chip1/metric3", true));  // throws
    metrics.push_back(std::make_unique<ThrowingMockMetric>("tray0/chip1/metric4", true));  // throws
    metrics.push_back(std::make_unique<ThrowingMockMetric>("tray0/chip2/metric5", false));

    size_t failed_count = update_metrics_with_exception_handling(metrics, null_cluster_, test_time_, "test");

    // Verify three failures were counted
    EXPECT_EQ(failed_count, 3);

    // Verify normal metrics still updated
    EXPECT_EQ(metrics[0]->get_update_count(), 1);
    EXPECT_EQ(metrics[4]->get_update_count(), 1);
}

TEST_F(ExceptionHandlingTest, AllMetricsSucceed_ReturnsZeroFailures) {
    // Create all non-throwing metrics
    std::vector<std::unique_ptr<ThrowingMockMetric>> metrics;
    metrics.push_back(std::make_unique<ThrowingMockMetric>("tray0/chip0/metric1", false));
    metrics.push_back(std::make_unique<ThrowingMockMetric>("tray0/chip0/metric2", false));
    metrics.push_back(std::make_unique<ThrowingMockMetric>("tray0/chip0/metric3", false));

    size_t failed_count = update_metrics_with_exception_handling(metrics, null_cluster_, test_time_, "test");

    // Verify no failures
    EXPECT_EQ(failed_count, 0);

    // Verify all metrics were updated
    for (const auto& metric : metrics) {
        EXPECT_EQ(metric->get_update_count(), 1);
    }
}

TEST_F(ExceptionHandlingTest, TelemetryPathString_FormatsCorrectly) {
    // Test the telemetry_path_string() method
    auto metric = std::make_unique<ThrowingMockMetric>("tray0/chip1/channel2/crcErrorCount", false);

    std::string path_str = metric->telemetry_path_string();

    EXPECT_EQ(path_str, "tray0/chip1/channel2/crcErrorCount");
}

TEST_F(ExceptionHandlingTest, TelemetryPathString_SingleComponent) {
    auto metric = std::make_unique<ThrowingMockMetric>("simple", false);

    std::string path_str = metric->telemetry_path_string();

    EXPECT_EQ(path_str, "simple");
}

TEST_F(ExceptionHandlingTest, EmptyMetricsList_ReturnsZeroFailures) {
    // Test with no metrics
    std::vector<std::unique_ptr<ThrowingMockMetric>> metrics;

    size_t failed_count = update_metrics_with_exception_handling(metrics, null_cluster_, test_time_, "test");

    EXPECT_EQ(failed_count, 0);
}

}  // namespace
