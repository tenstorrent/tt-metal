// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <telemetry/metric.hpp>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
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
        std::vector<std::string> components;
        std::string_view remaining = path_;

        while (!remaining.empty()) {
            size_t pos = remaining.find('/');
            if (pos == std::string_view::npos) {
                components.emplace_back(remaining);
                break;
            }
            if (pos > 0) {
                components.emplace_back(remaining.substr(0, pos));
            }
            remaining.remove_prefix(pos + 1);
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
    std::vector<std::unique_ptr<ThrowingMockMetric>> metrics;
    metrics.push_back(std::make_unique<ThrowingMockMetric>("tray0/chip0/metric1", true));  // throws
    metrics.push_back(std::make_unique<ThrowingMockMetric>("tray0/chip1/metric2", false));

    size_t failed_count = update_metrics_with_exception_handling(metrics, null_cluster_, test_time_, "test");

    EXPECT_EQ(failed_count, 1);
    EXPECT_EQ(metrics[0]->get_update_count(), 0);  // This one threw, so no update
    EXPECT_EQ(metrics[1]->get_update_count(), 1);  // This one succeeded despite the previous exception
}

TEST_F(ExceptionHandlingTest, MultipleExceptions_TracksAllFailures) {
    std::vector<std::unique_ptr<ThrowingMockMetric>> metrics;
    metrics.push_back(std::make_unique<ThrowingMockMetric>("tray0/chip0/metric1", true));  // throws
    metrics.push_back(std::make_unique<ThrowingMockMetric>("tray0/chip1/metric2", true));  // throws

    size_t failed_count = update_metrics_with_exception_handling(metrics, null_cluster_, test_time_, "test");

    EXPECT_EQ(failed_count, 2);
    EXPECT_EQ(metrics[0]->get_update_count(), 0);
    EXPECT_EQ(metrics[1]->get_update_count(), 0);
}

TEST_F(ExceptionHandlingTest, AllMetricsSucceed_ReturnsZeroFailures) {
    std::vector<std::unique_ptr<ThrowingMockMetric>> metrics;
    metrics.push_back(std::make_unique<ThrowingMockMetric>("tray0/chip0/metric1", false));
    metrics.push_back(std::make_unique<ThrowingMockMetric>("tray0/chip1/metric2", false));

    size_t failed_count = update_metrics_with_exception_handling(metrics, null_cluster_, test_time_, "test");

    EXPECT_EQ(failed_count, 0);
    EXPECT_EQ(metrics[0]->get_update_count(), 1);
    EXPECT_EQ(metrics[1]->get_update_count(), 1);
}

TEST_F(ExceptionHandlingTest, EmptyMetricsList_ReturnsZeroFailures) {
    // Test with no metrics
    std::vector<std::unique_ptr<ThrowingMockMetric>> metrics;

    size_t failed_count = update_metrics_with_exception_handling(metrics, null_cluster_, test_time_, "test");

    EXPECT_EQ(failed_count, 0);
}

}  // namespace
