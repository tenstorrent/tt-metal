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
#include <string_view>
#include <unordered_map>

#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>
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
        labels_changed_since_transmission_ = false;
    }

    virtual ~Metric() {
    }

    uint64_t timestamp() const {
        return timestamp_;
    }

    // Custom label support for Prometheus
    // Labels are key-value pairs that appear in Prometheus output alongside path-derived labels.
    //
    // IMPORTANT: Label keys must follow Prometheus naming conventions:
    //   - Must match [a-zA-Z_][a-zA-Z0-9_]*
    //   - Reserved prefixes "__" (double underscore) are for Prometheus internal use
    //   - Values are automatically escaped for special characters (\, ", \n)
    //   - Validation is enforced via assertion in debug builds
    //
    // Example usage:
    //   class MyMetric : public UIntMetric {
    //       void update(...) {
    //           set_value(42);
    //           set_label("user", "alice");
    //           set_label("process", "python3");
    //       }
    //   };
    //
    // Prometheus output:
    //   my_metric{hostname="...",device="0",user="alice",process="python3"} 42
    //
    // Labels can be set in constructor (static) or update() (dynamic).
    // Labels are mutable; when updated they are marked as changed so deltas can be transmitted.

private:
    // Helper: Check if character is ASCII letter (locale-independent)
    static constexpr bool is_ascii_alpha(char c) noexcept { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

    // Helper: Check if character is ASCII alphanumeric (locale-independent)
    static constexpr bool is_ascii_alnum(char c) noexcept { return is_ascii_alpha(c) || (c >= '0' && c <= '9'); }

    static bool is_valid_prometheus_label_key(std::string_view key) {
        if (key.empty()) {
            return false;
        }
        // First character must be ASCII letter or underscore
        // Use explicit ASCII checks to avoid locale-dependent behavior
        if (!is_ascii_alpha(key[0]) && key[0] != '_') {
            return false;
        }
        // Remaining characters must be ASCII alphanumeric or underscore
        for (size_t i = 1; i < key.size(); ++i) {
            if (!is_ascii_alnum(key[i]) && key[i] != '_') {
                return false;
            }
        }
        // Reject reserved "__" prefix (Prometheus internal use)
        if (key.size() >= 2 && key[0] == '_' && key[1] == '_') {
            return false;  // Reserved for Prometheus internal use
        }
        return true;
    }

public:
    void set_label(std::string_view key, std::string value) {
        // Validate in both debug and release builds for data integrity
        if (!is_valid_prometheus_label_key(key)) {
            log_error(
                tt::LogAlways,
                "Invalid Prometheus label key '{}': must match [a-zA-Z_][a-zA-Z0-9_]* and not start with '__'",
                key);
            return;  // Skip invalid labels
        }
        auto it = custom_labels_.find(std::string(key));
        if (it != custom_labels_.end()) {
            if (it->second != value) {
                it->second = std::move(value);
                labels_changed_since_transmission_ = true;
            }
        } else {
            custom_labels_.emplace(std::string(key), std::move(value));
            labels_changed_since_transmission_ = true;
        }
    }

    void set_labels(std::unordered_map<std::string, std::string> labels) {
        if (custom_labels_ != labels) {
            custom_labels_ = std::move(labels);
            labels_changed_since_transmission_ = true;
        }
    }

    const std::unordered_map<std::string, std::string>& labels() const { return custom_labels_; }

    bool labels_changed_since_transmission() const { return labels_changed_since_transmission_; }

protected:
    void set_timestamp_now() {
        timestamp_ =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count();
    }

    bool changed_since_transmission_ = false;
    bool labels_changed_since_transmission_ = false;
    uint64_t timestamp_ = 0;  // Unix timestamp in milliseconds, 0 = never set
    std::unordered_map<std::string, std::string> custom_labels_;
};

class BoolMetric: public Metric {
public:
    BoolMetric(MetricUnit metric_units = MetricUnit::UNITLESS) : Metric(metric_units) {}

    bool value() const {
        return value_;
    }

    void set_value(bool value) {
        changed_since_transmission_ = (value_ != value);
        value_ = value;
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
        changed_since_transmission_ = (value_ != value);
        value_ = value;
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
        changed_since_transmission_ = (value_ != value);
        value_ = value;
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
        changed_since_transmission_ = (value_ != value);
        value_ = std::move(value);
        set_timestamp_now();
    }

protected:
    std::string value_;
};
