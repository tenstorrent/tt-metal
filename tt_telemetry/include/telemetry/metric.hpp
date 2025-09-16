#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/metric.hpp
 *
 * Metric (i.e., telemetry point) types that we track. Various telemetry values derive from these.
 */

 #include <vector>
#include <chrono>
#include <string>
#include <unordered_map>

#include <third_party/umd/device/api/umd/device/cluster.hpp>

enum class MetricUnit : uint16_t {
    UNITLESS = 0,
    RESERVED_1 = 1,
    RESERVED_2 = 2,
    RESERVED_3 = 3,
    MEGAHERTZ = 4,
    WATTS = 5,
    MILLIVOLTS = 6,
    VOLTS = 7,
    REVOLUTIONS_PER_MINUTE = 8,
    AMPERES = 9,
    CELSIUS = 10
};

// Functions to convert MetricUnit enum to string labels
std::string metric_unit_to_display_label(MetricUnit unit);
std::string metric_unit_to_full_label(MetricUnit unit);

// Function to produce unit label maps for snapshots
std::unordered_map<uint16_t, std::string> create_metric_unit_display_label_map();
std::unordered_map<uint16_t, std::string> create_metric_unit_full_label_map();

class Metric {
public:
    const size_t id = 0;
    const MetricUnit units;

    Metric(size_t metric_unique_id, MetricUnit metric_units = MetricUnit::UNITLESS) :
        id(metric_unique_id), units(metric_units) {}

    virtual const std::vector<std::string> telemetry_path() const {
        return { "dummy", "metric", "someone", "forgot", "to", "implement", "telemetry", "path", "function" };
    }

    virtual void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    }

    bool changed_since_transmission() const {
        return changed_since_transmission_;
    }

    void mark_transmitted() {
        changed_since_transmission_ = false;
    }

    virtual ~Metric() {
    }

    uint64_t timestamp() const {
        return timestamp_;
    }

protected:
    bool changed_since_transmission_ = false;
    uint64_t timestamp_ = 0;  // Unix timestamp in milliseconds, 0 = never set
};

class BoolMetric: public Metric {
public:
    BoolMetric(size_t metric_unique_id, MetricUnit metric_units = MetricUnit::UNITLESS) :
        Metric(metric_unique_id, metric_units) {}

    bool value() const {
        return value_;
    }

protected:
    bool value_ = false;
};

class UIntMetric: public Metric {
public:
    UIntMetric(size_t metric_unique_id, MetricUnit metric_units = MetricUnit::UNITLESS) :
        Metric(metric_unique_id, metric_units) {}

    uint64_t value() const {
        return value_;
    }

protected:
    uint64_t value_ = 0;
};

class DoubleMetric : public Metric {
public:
    DoubleMetric(size_t metric_unique_id, MetricUnit metric_units = MetricUnit::UNITLESS) :
        Metric(metric_unique_id, metric_units) {}

    double value() const { return value_; }

protected:
    double value_ = 0.0;
};
