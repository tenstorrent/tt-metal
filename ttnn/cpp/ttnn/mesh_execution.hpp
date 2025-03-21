// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/decorators.hpp"  // Required for register_mesh_operation

namespace ttnn {

/**
 * Registers a device operation and automatically creates a mesh adapter for it.
 * This is a convenience function that allows registering operations with mesh support
 * without manually creating the adapter type.
 *
 * @tparam Name The fully qualified name for the operation
 * @tparam DeviceOp The device operation type
 * @return A registered operation with mesh support
 */
template <reflect::fixed_string Name, typename DeviceOp>
constexpr auto register_mesh_operation() {
    // Define the mesh-adapted operation type on the fly
    using MeshOpAdapter = MeshDeviceOperationAdapter<DeviceOp>;

    // Register the operation with the mesh adapter
    return decorators::register_operation<Name, MeshOpAdapter>();
}

}  // namespace ttnn
