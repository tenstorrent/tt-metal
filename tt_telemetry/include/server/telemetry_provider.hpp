#pragma once

/*
 * server/telemetry_provider.hpp
 *
 * Polls telemetry data on a periodic loop and sends to subscribers.
 */

#include <vector>

#include <server/telemetry_subscriber.hpp>

void run_telemetry_provider(std::vector<std::shared_ptr<TelemetrySubscriber>> subscribers);
