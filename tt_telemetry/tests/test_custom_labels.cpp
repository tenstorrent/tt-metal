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

// Test metric for custom label testing
class TestMetric : public UIntMetric {
private:
    std::string metric_name_;

public:
    TestMetric(std::string name) : UIntMetric(), metric_name_(std::move(name)) {}

    const std::vector<std::string> telemetry_path() const override { return {"device", "0", metric_name_}; }
};

// Test: Basic label setting and retrieval
TEST(CustomLabelsTest, BasicSetAndGet) {
    TestMetric metric("test_metric");

    // Initially no labels
    EXPECT_TRUE(metric.labels().empty());
    EXPECT_FALSE(metric.labels_changed_since_transmission());

    // Set a label
    metric.set_label("user", "alice");

    // Verify label exists
    EXPECT_EQ(metric.labels().size(), 1);
    EXPECT_EQ(metric.labels().at("user"), "alice");
    EXPECT_TRUE(metric.labels_changed_since_transmission());

    // Mark transmitted
    metric.mark_transmitted();
    EXPECT_FALSE(metric.labels_changed_since_transmission());
}

// Test: Multiple labels
TEST(CustomLabelsTest, MultipleLabels) {
    TestMetric metric("test_metric");

    metric.set_label("user", "alice");
    metric.set_label("process", "python3");
    metric.set_label("cmdline", "train.py");

    EXPECT_EQ(metric.labels().size(), 3);
    EXPECT_EQ(metric.labels().at("user"), "alice");
    EXPECT_EQ(metric.labels().at("process"), "python3");
    EXPECT_EQ(metric.labels().at("cmdline"), "train.py");
}

// Test: Label update (change value)
TEST(CustomLabelsTest, UpdateExistingLabel) {
    TestMetric metric("test_metric");

    metric.set_label("user", "alice");
    metric.mark_transmitted();

    // Update same key with different value
    metric.set_label("user", "bob");

    EXPECT_EQ(metric.labels().at("user"), "bob");
    EXPECT_TRUE(metric.labels_changed_since_transmission());
}

// Test: Label update with same value doesn't mark changed
TEST(CustomLabelsTest, SetSameValueDoesNotMarkChanged) {
    TestMetric metric("test_metric");

    metric.set_label("user", "alice");
    metric.mark_transmitted();
    EXPECT_FALSE(metric.labels_changed_since_transmission());

    // Set same value again
    metric.set_label("user", "alice");

    // Should NOT mark as changed
    EXPECT_FALSE(metric.labels_changed_since_transmission());
}

// Test: set_labels() bulk update
TEST(CustomLabelsTest, BulkSetLabels) {
    TestMetric metric("test_metric");

    std::unordered_map<std::string, std::string> labels = {{"user", "alice"}, {"process", "python3"}, {"pid", "12345"}};

    metric.set_labels(labels);

    EXPECT_EQ(metric.labels().size(), 3);
    EXPECT_EQ(metric.labels().at("user"), "alice");
    EXPECT_EQ(metric.labels().at("process"), "python3");
    EXPECT_EQ(metric.labels().at("pid"), "12345");
    EXPECT_TRUE(metric.labels_changed_since_transmission());
}

// Test: set_labels() with same map doesn't mark changed
TEST(CustomLabelsTest, BulkSetSameLabelsDoesNotMarkChanged) {
    TestMetric metric("test_metric");

    std::unordered_map<std::string, std::string> labels = {{"user", "alice"}};

    metric.set_labels(labels);
    metric.mark_transmitted();
    EXPECT_FALSE(metric.labels_changed_since_transmission());

    // Set same labels again
    metric.set_labels(labels);

    // Should NOT mark as changed
    EXPECT_FALSE(metric.labels_changed_since_transmission());
}

// Test: Valid label keys (ASCII alphanumeric + underscore)
TEST(CustomLabelsTest, ValidLabelKeys) {
    TestMetric metric("test_metric");

    // These should all succeed (no crash, no error)
    metric.set_label("user", "value");
    metric.set_label("process_name", "value");
    metric.set_label("_internal", "value");
    metric.set_label("key123", "value");
    metric.set_label("CamelCase", "value");

    EXPECT_EQ(metric.labels().size(), 5);
}

// Test: Invalid label keys are rejected
TEST(CustomLabelsTest, InvalidLabelKeysRejected) {
    TestMetric metric("test_metric");

    // These should be rejected (logged but not added)
    metric.set_label("", "value");            // Empty
    metric.set_label("123start", "value");    // Starts with digit
    metric.set_label("has-dash", "value");    // Contains dash
    metric.set_label("has.dot", "value");     // Contains dot
    metric.set_label("has space", "value");   // Contains space
    metric.set_label("__reserved", "value");  // Reserved prefix

    // No labels should have been added
    EXPECT_EQ(metric.labels().size(), 0);
}

// Test: TelemetrySnapshot label serialization
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

// Test: Snapshot merge with labels (fast path)
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

// Test: Label conflict detection
TEST(CustomLabelsTest, LabelConflictDetection) {
    TelemetrySnapshot base;
    base.uint_metrics["device/0/cpu"] = 85;
    base.metric_labels["device/0/cpu"] = {{"user", "alice"}};

    TelemetrySnapshot other;
    other.uint_metrics["device/0/cpu"] = 90;                  // Same path, different value
    other.metric_labels["device/0/cpu"] = {{"user", "bob"}};  // Conflicting label!

    base.merge_from(other, true);  // Validation path

    // Conflict should be rejected - original label preserved
    EXPECT_EQ(base.metric_labels["device/0/cpu"]["user"], "alice");
}

// Test: Non-conflicting labels merge successfully
TEST(CustomLabelsTest, NonConflictingLabelsMerge) {
    TelemetrySnapshot base;
    base.uint_metrics["device/0/cpu"] = 85;
    base.metric_labels["device/0/cpu"] = {{"user", "alice"}};

    TelemetrySnapshot other;
    other.uint_metrics["device/0/cpu"] = 90;
    other.metric_labels["device/0/cpu"] = {{"user", "alice"}};  // Same value - no conflict

    base.merge_from(other, true);  // Validation path

    // Should merge successfully
    EXPECT_EQ(base.metric_labels["device/0/cpu"]["user"], "alice");
    EXPECT_EQ(base.uint_metrics["device/0/cpu"], 90);  // Value updated
}

// Test: Labels only merged for successfully merged metrics
TEST(CustomLabelsTest, LabelsOnlyMergedForSuccessfulMetrics) {
    TelemetrySnapshot base;
    base.bool_metrics["device/0/status"] = true;

    TelemetrySnapshot other;
    other.uint_metrics["device/0/status"] = 42;  // Type conflict!
    other.metric_labels["device/0/status"] = {{"type", "cpu"}};

    base.merge_from(other, true);  // Validation path

    // Metric rejected due to type conflict, labels should also be rejected
    EXPECT_EQ(base.metric_labels.size(), 0);
}

// Test: Empty labels are handled correctly
TEST(CustomLabelsTest, EmptyLabelsHandled) {
    TestMetric metric("test_metric");

    // Metric with no labels
    EXPECT_TRUE(metric.labels().empty());

    // Setting and then clearing labels
    metric.set_label("user", "alice");
    metric.set_labels({});  // Clear all labels

    EXPECT_TRUE(metric.labels().empty());
    EXPECT_TRUE(metric.labels_changed_since_transmission());
}

// Test: Label change tracking independent of value changes
TEST(CustomLabelsTest, IndependentChangeTracking) {
    TestMetric metric("test_metric");

    metric.set_value(100);
    metric.set_label("user", "alice");
    metric.mark_transmitted();

    // Change only value
    metric.set_value(200);
    EXPECT_TRUE(metric.changed_since_transmission());
    EXPECT_FALSE(metric.labels_changed_since_transmission());

    metric.mark_transmitted();

    // Change only labels
    metric.set_label("user", "bob");
    EXPECT_FALSE(metric.changed_since_transmission());
    EXPECT_TRUE(metric.labels_changed_since_transmission());
}

}  // namespace
