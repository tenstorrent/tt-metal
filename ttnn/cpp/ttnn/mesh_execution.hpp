// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/attribute_customizer_helpers.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/decorators.hpp"  // Required for register_mesh_operation

namespace ttnn {

/**
 * Specialized version for registered operations (like ttnn::prim::dropout).
 *
 * This overload accepts operations created with ttnn::register_operation, allowing
 * direct passing of registered operations without wrapping them in lambdas.
 *
 * @tparam OpType The mesh device operation type that will be customized
 * @tparam F Type of the customizer function (deduced)
 * @tparam Name Name of the registered operation (deduced)
 * @tparam RegOp Registered operation type (deduced)
 * @tparam Args Argument types for the operation (deduced)
 *
 * @param customizer_func Function that constructs attributes per device
 * @param registered_op The registered operation to execute
 * @param args Arguments to pass to the operation
 * @return The result of executing the operation
 */
template <typename OpType, typename F, reflect::fixed_string Name, typename RegOp, typename... Args>
auto launch_mesh_workload(
    F&& customizer_func, const decorators::registered_operation_t<Name, RegOp>& registered_op, Args&&... args)
    -> decltype(registered_op(std::declval<Args>()...)) {
    // Adapt the simplified customizer to the full form expected by with_customizer
    auto adapter = [customizer_func = std::forward<F>(customizer_func)](
                       const typename OpType::operation_attributes_t& /*attrs*/,
                       const tt::tt_metal::distributed::MeshCoordinate& coord,
                       tt::tt_metal::distributed::MeshDevice* device) { return customizer_func(coord, device); };

    // Apply the customizer and execute the operation
    auto guard = with_customizer<OpType>(std::move(adapter));
    return registered_op(std::forward<Args>(args)...);
}

/**
 * Variant of launch_mesh_workload that accepts a lambda operation instead of a function pointer.
 *
 * This version is useful when the operation isn't a direct function call or requires additional setup.
 *
 * @tparam OpType The mesh device operation type that will be customized
 * @tparam F Type of the customizer function (deduced)
 * @tparam OpLambda Type of the operation lambda (deduced)
 *
 * @param customizer_func Function that constructs attributes per device
 * @param op_lambda Lambda function that executes the operation
 * @return The result of executing the operation
 */
template <typename OpType, typename F, typename OpLambda>
auto launch_mesh_workload(F&& customizer_func, OpLambda&& op_lambda) -> decltype(op_lambda()) {
    // Adapt the simplified customizer to the full form expected by with_customizer
    auto adapter = [customizer_func = std::forward<F>(customizer_func)](
                       const typename OpType::operation_attributes_t& /*attrs*/,
                       const tt::tt_metal::distributed::MeshCoordinate& coord,
                       tt::tt_metal::distributed::MeshDevice* device) { return customizer_func(coord, device); };

    // Apply the customizer and execute the operation
    auto guard = with_customizer<OpType>(std::move(adapter));
    return op_lambda();
}

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
