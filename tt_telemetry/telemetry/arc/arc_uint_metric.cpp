#include <telemetry/arc/arc_uint_metric.hpp>

#include <chrono>
#include <tt-metalium/assert.hpp>

/**************************************************************************************************
 ARCUintMetric Class
**************************************************************************************************/

ARCUintMetric::ARCUintMetric(
    size_t chip_id,
    std::shared_ptr<ARCTelemetryReader> reader,
    tt::umd::wormhole::TelemetryTag tag,
    const std::string& metric_name) :
    UIntMetric(chip_id),
    reader_(reader),
    wormhole_tag_(tag),
    blackhole_tag_()  // just some dummy tag we will never use
    ,
    metric_name_(metric_name) {
    TT_ASSERT(reader_ != nullptr, "ARCTelemetryReader cannot be null");
    TT_ASSERT(
        reader_->get_arch() == tt::ARCH::WORMHOLE_B0,
        "Reader architecture mismatch: expected Wormhole for chip {}",
        reader_->id);
    value_ = 0;
}

ARCUintMetric::ARCUintMetric(
    size_t chip_id,
    std::shared_ptr<ARCTelemetryReader> reader,
    tt::umd::blackhole::TelemetryTag tag,
    const std::string& metric_name) :
    UIntMetric(chip_id), reader_(reader), wormhole_tag_(), blackhole_tag_(tag), metric_name_(metric_name) {
    TT_ASSERT(reader_ != nullptr, "ARCTelemetryReader cannot be null");
    TT_ASSERT(
        reader_->get_arch() == tt::ARCH::BLACKHOLE,
        "Reader architecture mismatch: expected Blackhole for chip {}",
        reader_->id);
    value_ = 0;
}

const std::vector<std::string> ARCUintMetric::telemetry_path() const {
    // Start with the chip identifier path
    std::vector<std::string> path = reader_->id.telemetry_path();

    // Add the metric name
    path.push_back(metric_name_);

    return path;
}

void ARCUintMetric::update(const tt::Cluster& cluster) {
    uint64_t new_value = 0;

    // Read the appropriate telemetry value based on architecture
    tt::ARCH arch = reader_->get_arch();
    if (arch == tt::ARCH::WORMHOLE_B0) {
        new_value = static_cast<uint64_t>(reader_->read_value(wormhole_tag_));
    } else if (arch == tt::ARCH::BLACKHOLE) {
        new_value = static_cast<uint64_t>(reader_->read_value(blackhole_tag_));
    } else {
        TT_ASSERT(false, "Unsupported architecture for chip {}", reader_->id);
    }

    // Update the metric value and timestamp
    uint64_t old_value = value_;
    changed_since_transmission_ = new_value != old_value;
    value_ = new_value;
    timestamp_ =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
}
