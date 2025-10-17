// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/metric.cpp
 *
 * Implementation of MetricUnit enum functions and utility functions.
 */

#include <telemetry/metric.hpp>

std::string metric_unit_to_display_label(MetricUnit unit) {
    switch (unit) {
        case MetricUnit::UNITLESS: return "";
        case MetricUnit::RESERVED_1: return "<unknown>";
        case MetricUnit::RESERVED_2: return "<unknown>";
        case MetricUnit::RESERVED_3: return "<unknown>";
        case MetricUnit::MEGAHERTZ: return "MHz";
        case MetricUnit::WATTS: return "W";
        case MetricUnit::MILLIVOLTS: return "mV";
        case MetricUnit::VOLTS: return "V";
        case MetricUnit::REVOLUTIONS_PER_MINUTE: return "RPM";
        case MetricUnit::AMPERES: return "A";
        case MetricUnit::CELSIUS: return "°C";
        default: return "<unknown>";
    }
}

std::string metric_unit_to_full_label(MetricUnit unit) {
    switch (unit) {
        case MetricUnit::UNITLESS: return "";
        case MetricUnit::RESERVED_1: return "<unknown>";
        case MetricUnit::RESERVED_2: return "<unknown>";
        case MetricUnit::RESERVED_3: return "<unknown>";
        case MetricUnit::MEGAHERTZ: return "Megahertz";
        case MetricUnit::WATTS: return "Watts";
        case MetricUnit::MILLIVOLTS: return "Millivolts";
        case MetricUnit::VOLTS: return "Volts";
        case MetricUnit::REVOLUTIONS_PER_MINUTE: return "Revolutions Per Minute";
        case MetricUnit::AMPERES: return "Amperes";
        case MetricUnit::CELSIUS: return "Celsius";
        default: return "<unknown>";
    }
}

std::unordered_map<uint16_t, std::string> create_metric_unit_display_label_map() {
    std::unordered_map<uint16_t, std::string> map;

    // Add all defined metric units
    map[static_cast<uint16_t>(MetricUnit::UNITLESS)] = metric_unit_to_display_label(MetricUnit::UNITLESS);
    map[static_cast<uint16_t>(MetricUnit::RESERVED_1)] = metric_unit_to_display_label(MetricUnit::RESERVED_1);
    map[static_cast<uint16_t>(MetricUnit::RESERVED_2)] = metric_unit_to_display_label(MetricUnit::RESERVED_2);
    map[static_cast<uint16_t>(MetricUnit::RESERVED_3)] = metric_unit_to_display_label(MetricUnit::RESERVED_3);
    map[static_cast<uint16_t>(MetricUnit::MEGAHERTZ)] = metric_unit_to_display_label(MetricUnit::MEGAHERTZ);
    map[static_cast<uint16_t>(MetricUnit::WATTS)] = metric_unit_to_display_label(MetricUnit::WATTS);
    map[static_cast<uint16_t>(MetricUnit::MILLIVOLTS)] = metric_unit_to_display_label(MetricUnit::MILLIVOLTS);
    map[static_cast<uint16_t>(MetricUnit::VOLTS)] = metric_unit_to_display_label(MetricUnit::VOLTS);
    map[static_cast<uint16_t>(MetricUnit::REVOLUTIONS_PER_MINUTE)] =
        metric_unit_to_display_label(MetricUnit::REVOLUTIONS_PER_MINUTE);
    map[static_cast<uint16_t>(MetricUnit::AMPERES)] = metric_unit_to_display_label(MetricUnit::AMPERES);
    map[static_cast<uint16_t>(MetricUnit::CELSIUS)] = metric_unit_to_display_label(MetricUnit::CELSIUS);

    return map;
}

std::unordered_map<uint16_t, std::string> create_metric_unit_full_label_map() {
    std::unordered_map<uint16_t, std::string> map;

    // Add all defined metric units
    map[static_cast<uint16_t>(MetricUnit::UNITLESS)] = metric_unit_to_full_label(MetricUnit::UNITLESS);
    map[static_cast<uint16_t>(MetricUnit::RESERVED_1)] = metric_unit_to_full_label(MetricUnit::RESERVED_1);
    map[static_cast<uint16_t>(MetricUnit::RESERVED_2)] = metric_unit_to_full_label(MetricUnit::RESERVED_2);
    map[static_cast<uint16_t>(MetricUnit::RESERVED_3)] = metric_unit_to_full_label(MetricUnit::RESERVED_3);
    map[static_cast<uint16_t>(MetricUnit::MEGAHERTZ)] = metric_unit_to_full_label(MetricUnit::MEGAHERTZ);
    map[static_cast<uint16_t>(MetricUnit::WATTS)] = metric_unit_to_full_label(MetricUnit::WATTS);
    map[static_cast<uint16_t>(MetricUnit::MILLIVOLTS)] = metric_unit_to_full_label(MetricUnit::MILLIVOLTS);
    map[static_cast<uint16_t>(MetricUnit::VOLTS)] = metric_unit_to_full_label(MetricUnit::VOLTS);
    map[static_cast<uint16_t>(MetricUnit::REVOLUTIONS_PER_MINUTE)] =
        metric_unit_to_full_label(MetricUnit::REVOLUTIONS_PER_MINUTE);
    map[static_cast<uint16_t>(MetricUnit::AMPERES)] = metric_unit_to_full_label(MetricUnit::AMPERES);
    map[static_cast<uint16_t>(MetricUnit::CELSIUS)] = metric_unit_to_full_label(MetricUnit::CELSIUS);

    return map;
}
