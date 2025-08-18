#pragma once

/*
 * telemetry/telemetry_provider.hpp
 *
 * Polls telemetry data on a periodic loop and sends to subscribers.
 */

#include <vector>

#include <telemetry/telemetry_subscriber.hpp>

void run_telemetry_provider(std::vector<std::shared_ptr<TelemetrySubscriber>> subscribers);
