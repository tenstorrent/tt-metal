// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"

namespace ttnn::prim::conv2d_new_detail {

// MeshWorkloadFactoryConcept factory for height-sharded and block-sharded conv2d.
// Uses ProgramDescriptor internally for clean declarative program construction,
// while handling config tensor lifecycle and dynamic CB patching in the
// mesh workload methods.
struct Conv2dShardedDescriptorFactory {
    struct AddressSlot {
        uint32_t kernel_handle;
        CoreCoord core;
        uint32_t arg_index;
        uint16_t buffer_id;
    };

    struct CBSlot {
        tt::tt_metal::CBHandle cb_handle;
        uint16_t buffer_id;
    };

    struct shared_variables_t {
        std::vector<AddressSlot> address_slots;
        std::vector<CBSlot> cb_slots;
        tt::tt_metal::DeviceStorage conv_reader_indices_storage;
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const Conv2dParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const Conv2dInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const Conv2dParams& operation_attributes,
        const Conv2dInputs& tensor_args,
        Tensor& tensor_return_value);

    // Internal: build the declarative program descriptor.
    // config_tensor_buffer is the device buffer for sliding window reader indices,
    // created in create_mesh_workload before this is called.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const Conv2dParams& operation_attributes,
        const Conv2dInputs& tensor_args,
        Tensor& output,
        tt::tt_metal::Buffer* config_tensor_buffer);
};

}  // namespace ttnn::prim::conv2d_new_detail
