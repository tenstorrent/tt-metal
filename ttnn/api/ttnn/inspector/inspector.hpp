// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace ttnn {
namespace inspector {

/**
 * @brief Registers the TTNN Inspector RPC channel with the Inspector RPC server.
 *
 * This function enables TTNN runtime state to be queried through the Inspector interface.
 * It should be called during system or application initialization, after the Inspector RPC server
 * has been started and is ready to accept channel registrations.
 *
 * Prerequisites:
 *   - The Inspector RPC server must be running and accessible.
 *   - Any required TTNN runtime components should be initialized prior to calling this function.
 */
void register_inspector_rpc();

}  // namespace inspector
}  // namespace ttnn
