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
    const std::string& metric_name,
    uint32_t mask) :
    UIntMetric(chip_id),
    reader_(reader),
    wormhole_tag_(tag),
    blackhole_tag_(),  // just some dummy tag we will never use
    metric_name_(metric_name),
    mask_(mask) {
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
    const std::string& metric_name,
    uint32_t mask) :
    UIntMetric(chip_id), reader_(reader), wormhole_tag_(), blackhole_tag_(tag), metric_name_(metric_name), mask_(mask) {
    TT_ASSERT(reader_ != nullptr, "ARCTelemetryReader cannot be null");
    TT_ASSERT(
        reader_->get_arch() == tt::ARCH::BLACKHOLE,
        "Reader architecture mismatch: expected Blackhole for chip {}",
        reader_->id);
    value_ = 0;
}

ARCUintMetric::ARCUintMetric(
    size_t chip_id, std::shared_ptr<ARCTelemetryReader> reader, CommonTelemetryTag common_metric) :
    UIntMetric(chip_id), reader_(reader), wormhole_tag_(), blackhole_tag_(), mask_(0xffffffff) {
    TT_ASSERT(reader_ != nullptr, "ARCTelemetryReader cannot be null");

    // Set metric name and tags based on common metric type
    switch (common_metric) {
        case CommonTelemetryTag::AICLK:
            metric_name_ = "AIClock";
            wormhole_tag_ = tt::umd::wormhole::TelemetryTag::AICLK;
            blackhole_tag_ = tt::umd::blackhole::TelemetryTag::AICLK;
            mask_ = 0xffff;  // 16-bit value
            break;
        case CommonTelemetryTag::AXICLK:
            metric_name_ = "AXIClock";
            wormhole_tag_ = tt::umd::wormhole::TelemetryTag::AXICLK;
            blackhole_tag_ = tt::umd::blackhole::TelemetryTag::AXICLK;
            break;
        case CommonTelemetryTag::ARCCLK:
            metric_name_ = "ARCClock";
            wormhole_tag_ = tt::umd::wormhole::TelemetryTag::ARCCLK;
            blackhole_tag_ = tt::umd::blackhole::TelemetryTag::ARCCLK;
            break;
        case CommonTelemetryTag::FAN_SPEED:
            metric_name_ = "FanSpeed";
            wormhole_tag_ = tt::umd::wormhole::TelemetryTag::FAN_SPEED;
            blackhole_tag_ = tt::umd::blackhole::TelemetryTag::FAN_SPEED;
            break;
        case CommonTelemetryTag::TDP:
            metric_name_ = "TDP";
            wormhole_tag_ = tt::umd::wormhole::TelemetryTag::TDP;
            blackhole_tag_ = tt::umd::blackhole::TelemetryTag::TDP;
            mask_ = 0xffff;  // 16-bit value
            break;
        case CommonTelemetryTag::TDC:
            metric_name_ = "TDC";
            wormhole_tag_ = tt::umd::wormhole::TelemetryTag::TDC;
            blackhole_tag_ = tt::umd::blackhole::TelemetryTag::TDC;
            mask_ = 0xffff;  // 16-bit value
            break;
        case CommonTelemetryTag::VCORE:
            metric_name_ = "VCore";
            wormhole_tag_ = tt::umd::wormhole::TelemetryTag::VCORE;
            blackhole_tag_ = tt::umd::blackhole::TelemetryTag::VCORE;
            break;
        case CommonTelemetryTag::ASIC_TEMPERATURE:
            metric_name_ = "ASICTemperature";
            wormhole_tag_ = tt::umd::wormhole::TelemetryTag::ASIC_TEMPERATURE;
            blackhole_tag_ = tt::umd::blackhole::TelemetryTag::ASIC_TEMPERATURE;
            mask_ = (reader_->get_arch() == tt::ARCH::WORMHOLE_B0) ? 0xffff : 0xffffffff;
            break;
        case CommonTelemetryTag::BOARD_TEMPERATURE:
            metric_name_ = "BoardTemperature";
            wormhole_tag_ = tt::umd::wormhole::TelemetryTag::BOARD_TEMPERATURE;
            blackhole_tag_ = tt::umd::blackhole::TelemetryTag::BOARD_TEMPERATURE;
            mask_ = (reader_->get_arch() == tt::ARCH::WORMHOLE_B0) ? 0xffff : 0xffffffff;
            break;
        default: TT_ASSERT(false, "Unknown CommonTelemetryTag type for chip {}", reader_->id);
    }

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
    uint32_t raw_value = 0;
    tt::ARCH arch = reader_->get_arch();
    if (arch == tt::ARCH::WORMHOLE_B0) {
        raw_value = reader_->read_value(wormhole_tag_);
    } else if (arch == tt::ARCH::BLACKHOLE) {
        raw_value = reader_->read_value(blackhole_tag_);
    } else {
        TT_ASSERT(false, "Unsupported architecture for chip {}", reader_->id);
    }

    // Apply mask to get the final value
    new_value = static_cast<uint64_t>(raw_value & mask_);

    // Update the metric value and timestamp
    uint64_t old_value = value_;
    changed_since_transmission_ = new_value != old_value;
    value_ = new_value;
    timestamp_ =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
}
