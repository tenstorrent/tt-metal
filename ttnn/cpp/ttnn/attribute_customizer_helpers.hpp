// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include "ttnn/mesh_device_operation_adapter.hpp"

namespace ttnn {

/**
 * RAII wrapper for attribute customizers that automatically resets the customizer when it goes out of scope.
 * This makes it easy to apply a dynamic customizer for a specific scope without worrying about cleanup.
 *
 * @tparam MeshDeviceOpT The mesh device operation type that will use the customizer
 * @tparam AttributeType The attribute type being customized
 */
template <typename MeshDeviceOpT, typename AttributeType>
class ScopedAttributeCustomizer {
public:
    /**
     * Creates a scoped attribute customizer from any callable that satisfies the AttributeCustomizerCallable concept.
     *
     * @tparam F The callable type (typically a lambda)
     * @param func The customization function
     */
    template <AttributeCustomizerCallable<AttributeType> F>
    ScopedAttributeCustomizer(F&& func) {
        MeshDeviceOpT::set_attribute_customizer(make_attribute_customizer<AttributeType>(std::forward<F>(func)));
    }

    /**
     * Creates a scoped attribute customizer from an existing customizer instance.
     *
     * @param customizer A shared pointer to an AttributeCustomizerBase
     */
    ScopedAttributeCustomizer(std::shared_ptr<AttributeCustomizerBase<AttributeType>> customizer) {
        MeshDeviceOpT::set_attribute_customizer(std::move(customizer));
    }

    /**
     * Destructor - automatically resets the customizer when the object goes out of scope.
     */
    ~ScopedAttributeCustomizer() { MeshDeviceOpT::reset_attribute_customizer(); }

    // Prevent copying and moving
    ScopedAttributeCustomizer(const ScopedAttributeCustomizer&) = delete;
    ScopedAttributeCustomizer& operator=(const ScopedAttributeCustomizer&) = delete;
    ScopedAttributeCustomizer(ScopedAttributeCustomizer&&) = delete;
    ScopedAttributeCustomizer& operator=(ScopedAttributeCustomizer&&) = delete;
};

/**
 * Helper function that creates a scoped attribute customizer with a simpler interface.
 * The customizer is automatically cleaned up when it goes out of scope.
 *
 * @tparam OperationType The operation to apply the customizer to
 * @tparam F Type of the customizer function (automatically deduced)
 * @param func The customizer function that modifies attributes per device
 * @return A scoped guard that applies and cleans up the customizer
 */
template <typename OperationType, typename F>
    requires AttributeCustomizerCallable<F, typename OperationType::operation_attributes_t>
[[nodiscard]] auto with_customizer(F&& func) {
    using attribute_type = typename OperationType::operation_attributes_t;
    return ScopedAttributeCustomizer<OperationType, attribute_type>(std::forward<F>(func));
}

}  // namespace ttnn
