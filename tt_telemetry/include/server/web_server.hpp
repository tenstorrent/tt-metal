#pragma once

#include <future>
#include <utility>

#include <server/telemetry_subscriber.hpp>

/*
 * server/web_server.hpp
 *
 * Built-in web server for broadcasting telemetry data.
 */

std::pair<std::future<bool>, std::shared_ptr<TelemetrySubscriber>> run_web_server(uint16_t port);
