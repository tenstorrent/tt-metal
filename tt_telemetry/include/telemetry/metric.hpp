#pragma once

/*
 * telemetry/metric.hpp
 *
 * Metric (i.e., telemetry point) types that we track. Various telemetry values derive from these.
 */

 #include <vector>
 #include <chrono>

 #include <llrt/tt_cluster.hpp>

class Metric {
public:
    const size_t id = 0;

    Metric(size_t metric_unique_id)
        : id(metric_unique_id)
    {
    }

    virtual const std::vector<std::string> telemetry_path() const {
        return { "dummy", "metric", "someone", "forgot", "to", "implement", "telemetry", "path", "function" };
    }

    virtual void update(const tt::Cluster &cluster) {
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
    BoolMetric(size_t metric_unique_id)
        : Metric(metric_unique_id)
    {
    }

    bool value() const {
        return value_;
    }

protected:
    bool value_ = false;
};

class UIntMetric: public Metric {
public:
    UIntMetric(size_t metric_unique_id)
        : Metric(metric_unique_id)
    {
    }

    uint64_t value() const {
        return value_;
    }

protected:
    uint64_t value_ = 0;
};
