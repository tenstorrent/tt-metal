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

#include <fmt/ranges.h>
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
    const MetricUnit units;

    Metric(MetricUnit metric_units = MetricUnit::UNITLESS) : units(metric_units) {}

    virtual const std::vector<std::string> telemetry_path() const {
        return { "dummy", "metric", "someone", "forgot", "to", "implement", "telemetry", "path", "function" };
    }

    // Get telemetry path as a slash-separated string
    std::string telemetry_path_string() const { return fmt::format("{}", fmt::join(telemetry_path(), "/")); }

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
    void set_timestamp_now() {
        timestamp_ =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count();
    }

    bool changed_since_transmission_ = false;
    uint64_t timestamp_ = 0;  // Unix timestamp in milliseconds, 0 = never set
};

class BoolMetric: public Metric {
public:
    BoolMetric(MetricUnit metric_units = MetricUnit::UNITLESS) : Metric(metric_units) {}

    bool value() const {
        return value_;
    }

    void set_value(bool value) {
        value_ = value;
        changed_since_transmission_ = true;
        set_timestamp_now();
    }

protected:
    bool value_ = false;
};

class UIntMetric: public Metric {
public:
    UIntMetric(MetricUnit metric_units = MetricUnit::UNITLESS) : Metric(metric_units) {}

    uint64_t value() const {
        return value_;
    }

    void set_value(uint64_t value) {
        value_ = value;
        changed_since_transmission_ = true;
        set_timestamp_now();
    }

protected:
    uint64_t value_ = 0;
};

class DoubleMetric : public Metric {
public:
    DoubleMetric(MetricUnit metric_units = MetricUnit::UNITLESS) : Metric(metric_units) {}

    double value() const { return value_; }

    void set_value(double value) {
        value_ = value;
        changed_since_transmission_ = true;
        set_timestamp_now();
    }

protected:
    double value_ = 0.0;
};

class StringMetric : public Metric {
public:
    StringMetric(MetricUnit metric_units = MetricUnit::UNITLESS) : Metric(metric_units) {}

    std::string_view value() const { return value_; }

    void set_value(std::string value) {
        value_ = std::move(value);
        changed_since_transmission_ = true;
        set_timestamp_now();
    }

protected:
    std::string value_;
};
