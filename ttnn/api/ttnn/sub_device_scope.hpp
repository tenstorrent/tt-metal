// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/sub_device_types.hpp>

#include "ttnn/common/guard.hpp"

namespace ttnn {

// Runs every operation the calling thread launches on one sub-device of its device, for the lifetime of the returned
// guard. Scopes nest, and the innermost applies. Inside a scope, an operation that does not record the sub-device it
// runs on (see ttnn::AssignedSubDevice) is refused. A scope keeps the sub-device manager its first operation ran under:
// once another is loaded, its operations are refused, since the same id can name other cores. Operations are checked
// as each one launches, so a composite operation can be refused partway; what it launched before is not undone (see
// ttnn::CompositeSubDevice).
[[nodiscard]] ScopeGuard with_sub_device(tt::tt_metal::SubDeviceId sub_device_id);

// The same scope, opened and closed explicitly. Push returns a token that closes it; scopes must close innermost first.
[[nodiscard]] std::uint64_t push_sub_device_for_thread(tt::tt_metal::SubDeviceId sub_device_id);
void pop_sub_device_for_thread(std::uint64_t token);

// Whether the calling thread runs inside a sub-device scope, or a composite operation's sub-device.
bool in_sub_device_scope();

}  // namespace ttnn
