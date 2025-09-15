#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * server/websocket_test_client.hpp
 *
 * WebSocket test client for testing WebSocket server functionality.
 */

#include <cstdint>

/**
 * Run a WebSocket test client that connects to the specified port.
 * The client will run for 30 seconds, sending periodic responses to the server.
 *
 * @param port The port to connect to
 */
void run_websocket_test_client(uint16_t port);
