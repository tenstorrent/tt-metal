// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>
#include <ttnn/device_operation.hpp>
#include <ttnn/distributed/types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::experimental::subdevice_sync {

enum class SubdeviceSyncMode : uint8_t { Signal = 0, Wait = 1 };

// One-core program used for in-trace cross-sub-device hand-offs (see subdevice_sync.hpp). The same program runs on
// every chip of the mesh (chip-local semaphore, no cross-chip traffic).
struct SubdeviceSyncDeviceOperation {
    struct operation_attributes_t {
        SubdeviceSyncMode mode;
        uint32_t semaphore_address;  // dynamic (not hashed)
        CoreCoord core;              // logical core the kernel runs on (signal core / wait core)
        CoreCoord target_core;       // Signal: logical core holding the semaphore word (waiter core); dynamic
        uint32_t value;              // dynamic
        uint32_t timeout_cycles;     // Wait only; 0 = spin forever; dynamic
        uint8_t sub_device_id;
        ttnn::MeshDevice* mesh_device;

        // Semaphore address, target core, value and timeout are runtime args re-applied on every cache hit, so one
        // cached program per (mode, core, sub-device) serves every hand-off edge.
        static constexpr auto attribute_names = std::forward_as_tuple("mode", "core", "sub_device_id", "mesh_device");
        auto attribute_values() const { return std::forward_as_tuple(mode, core, sub_device_id, mesh_device); }
    };

    struct tensor_args_t {};

    // The infrastructure does not support void returns; no tensors are produced.
    using tensor_return_value_t = std::vector<ttnn::Tensor>;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;

    struct SubdeviceSyncProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            tt::tt_metal::Program& program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    using program_factory_t = std::variant<SubdeviceSyncProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::subdevice_sync

namespace ttnn::prim {

void subdevice_sync(
    ttnn::MeshDevice& mesh_device,
    ttnn::operations::experimental::subdevice_sync::SubdeviceSyncMode mode,
    uint32_t semaphore_address,
    const CoreCoord& core,
    const CoreCoord& target_core,
    uint32_t value,
    uint32_t timeout_cycles,
    tt::tt_metal::SubDeviceId sub_device_id);

}  // namespace ttnn::prim
