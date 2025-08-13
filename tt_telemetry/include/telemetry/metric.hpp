#pragma once

/*
 * telemetry/metric.hpp
 *
 * Metric (i.e., telemetry point) types that we track. Various telemetry values derive from these.
 */

 #include <vector>

 #include <llrt/tt_cluster.hpp>

class Metric {
public:
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

protected:
    bool changed_since_transmission_ = false;
};

class BoolMetric: public Metric {
public:
    bool value() const {
        return value_;
    }

protected:
    bool value_ = false;
};