// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <telemetry/metric.hpp>
#include <telemetry/telemetry_snapshot.hpp>
#include <memory>
#include <string>
#include <unordered_map>

namespace {

// Test metric with immutable custom labels
class TestMetricWithLabels : public UIntMetric {
private:
    std::string metric_name_;
    std::unordered_map<std::string, std::string> labels_;

public:
    TestMetricWithLabels(std::string name, std::unordered_map<std::string, std::string> labels = {}) :
        UIntMetric(), metric_name_(std::move(name)), labels_(std::move(labels)) {}

    const std::vector<std::string> telemetry_path() const override { return {"device", "0", metric_name_}; }

    std::unordered_map<std::string, std::string> labels() const override { return labels_; }
};

// Test metric with no labels (uses default implementation)
class TestMetricNoLabels : public UIntMetric {
private:
    std::string metric_name_;

public:
    TestMetricNoLabels(std::string name) : UIntMetric(), metric_name_(std::move(name)) {}

    const std::vector<std::string> telemetry_path() const override { return {"device", "0", metric_name_}; }
    // Uses default labels() implementation which returns empty map
};

// Test: Basic label retrieval
TEST(CustomLabelsTest, BasicLabels) {
    TestMetricWithLabels metric("test_metric", {{"user", "alice"}, {"process", "python3"}, {"pid", "12345"}});

    auto labels = metric.labels();
    EXPECT_EQ(labels.size(), 3);
    EXPECT_EQ(labels.at("user"), "alice");
    EXPECT_EQ(labels.at("process"), "python3");
    EXPECT_EQ(labels.at("pid"), "12345");
}

// Test: Empty labels (default implementation)
TEST(CustomLabelsTest, NoLabels) {
    TestMetricNoLabels metric("test_metric");

    auto labels = metric.labels();
    EXPECT_TRUE(labels.empty());
}

// Test: TelemetrySnapshot label storage
TEST(CustomLabelsTest, SnapshotLabelStorage) {
    TelemetrySnapshot snapshot;

    // Add metrics with labels
    snapshot.uint_metrics["device/0/cpu"] = 85;
    snapshot.metric_labels["device/0/cpu"] = {{"user", "alice"}, {"process", "python3"}};

    // Verify labels stored correctly
    EXPECT_EQ(snapshot.metric_labels.size(), 1);
    EXPECT_EQ(snapshot.metric_labels["device/0/cpu"].size(), 2);
    EXPECT_EQ(snapshot.metric_labels["device/0/cpu"]["user"], "alice");
    EXPECT_EQ(snapshot.metric_labels["device/0/cpu"]["process"], "python3");
}

// Test: Snapshot merge with labels from different metrics (fast path)
TEST(CustomLabelsTest, SnapshotMergeFast) {
    TelemetrySnapshot base;
    base.uint_metrics["device/0/cpu"] = 85;
    base.metric_labels["device/0/cpu"] = {{"user", "alice"}};

    TelemetrySnapshot other;
    other.uint_metrics["device/1/cpu"] = 92;
    other.metric_labels["device/1/cpu"] = {{"user", "bob"}};

    base.merge_from(other, false);  // Fast path

    // Both metrics and labels should be merged
    EXPECT_EQ(base.uint_metrics.size(), 2);
    EXPECT_EQ(base.metric_labels.size(), 2);
    EXPECT_EQ(base.metric_labels["device/0/cpu"]["user"], "alice");
    EXPECT_EQ(base.metric_labels["device/1/cpu"]["user"], "bob");
}

// Test: Orphaned labels rejected in validation path
TEST(CustomLabelsTest, OrphanedLabelsRejected) {
    TelemetrySnapshot base;
    base.uint_metrics["device/0/cpu"] = 85;

    TelemetrySnapshot other;
    // No metric, but has labels (orphaned)
    other.metric_labels["device/1/cpu"] = {{"user", "bob"}};

    base.merge_from(other, true);  // Validation path

    // Orphaned labels should NOT be merged
    EXPECT_EQ(base.metric_labels.size(), 0);
}

// Test: Immutable labels use first-wins semantics on merge
// This tests that insert() preserves existing labels (doesn't overwrite)
TEST(CustomLabelsTest, ImmutableLabelsFirstWins) {
    TelemetrySnapshot base;
    base.uint_metrics["device/0/cpu"] = 85;
    base.metric_labels["device/0/cpu"] = {{"user", "alice"}};

    TelemetrySnapshot other;
    other.uint_metrics["device/0/cpu"] = 90;                  // Same path, different value
    other.metric_labels["device/0/cpu"] = {{"user", "bob"}};  // Different labels (shouldn't happen, but test semantics)

    base.merge_from(other, true);  // Validation path

    // With immutable labels using insert(), existing labels are preserved (first-wins)
    EXPECT_EQ(base.metric_labels["device/0/cpu"]["user"], "alice");
    EXPECT_EQ(base.uint_metrics["device/0/cpu"], 90);  // Value updated
}

// Test: Labels rejected when metric rejected due to type conflict
TEST(CustomLabelsTest, LabelsRejectedWithTypeConflict) {
    TelemetrySnapshot base;
    base.bool_metrics["device/0/status"] = true;

    TelemetrySnapshot other;
    other.uint_metrics["device/0/status"] = 42;  // Type conflict!
    other.metric_labels["device/0/status"] = {{"type", "cpu"}};

    base.merge_from(other, true);  // Validation path

    // Metric rejected due to type conflict, labels should also be rejected
    EXPECT_EQ(base.metric_labels.size(), 0);
}

}  // namespace
