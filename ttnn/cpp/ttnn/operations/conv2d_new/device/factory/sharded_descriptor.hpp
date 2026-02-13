// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"

namespace ttnn::prim::conv2d_new_detail {

// ProgramDescriptorFactoryConcept factory for height-sharded and block-sharded conv2d.
//
// Uses the optional prepare_resources hook to create the sliding window config
// tensor (a device-side allocation).  The framework's DescriptorMeshWorkloadFactoryAdapter
// handles all cache-hit dispatch: buffer address patching, dynamic CB patching,
// and resource lifetime management.
struct Conv2dShardedDescriptorFactory {
    // Creates the sliding window config tensor (device-side allocation).
    // Called once on cache miss; the returned DeviceStorage is kept alive across
    // cache hits by the framework.
    static tt::tt_metal::DeviceStorage prepare_resources(
        const Conv2dParams& operation_attributes, const Conv2dInputs& tensor_args, Tensor& tensor_return_value);

    // Builds the declarative ProgramDescriptor.
    // resources holds the config tensor buffer from prepare_resources.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const Conv2dParams& operation_attributes,
        const Conv2dInputs& tensor_args,
        Tensor& output,
        tt::tt_metal::DeviceStorage& resources);
};

}  // namespace ttnn::prim::conv2d_new_detail
