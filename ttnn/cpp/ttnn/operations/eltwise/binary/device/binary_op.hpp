// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <variant>

#include "binary_op_type.hpp"
#include "broadcast_height_and_width_multi_core_program_factory.hpp"
#include "broadcast_height_multi_core_program_factory.hpp"
#include "broadcast_width_multi_core_program_factory.hpp"
#include "element_wise_multi_core_program_factory.hpp"
#include "tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_eager/tensor/host_buffer/functions.hpp"
#include "tt_eager/tensor/tensor_utils.hpp"
#include "tt_eager/tt_dnn/op_library/compute_kernel_config.hpp"
#include "tt_eager/tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::binary {

constexpr uint8_t DefaultQueueId = 0;

struct Binary {
    struct operation_attributes_t {
        BinaryOpType binary_op_type;
        bool in_place;
        const std::optional<FusedActivations> activations;
        const MemoryConfig memory_config;
        const DataType dtype;
        std::optional<DeviceComputeKernelConfig> compute_kernel_config;

        static constexpr auto attribute_names = std::forward_as_tuple(
            "binary_op_type", "in_place", "activations", "memory_config", "dtype", "compute_kernel_config");
        const auto attribute_values() const {
            return std::forward_as_tuple(
                this->binary_op_type,
                this->in_place,
                this->activations,
                this->memory_config,
                this->dtype,
                this->compute_kernel_config);
        }
    };
    struct tensor_args_t {
        const Tensor& input_tensor_a;
        const Tensor& input_tensor_b;
        std::optional<Tensor> output_tensor;

        static constexpr auto attribute_names =
            std::forward_as_tuple("input_tensor_a", "input_tensor_b", "output_tensor");
        const auto attribute_values() const {
            return std::forward_as_tuple(this->input_tensor_a, this->input_tensor_b, this->output_tensor);
        }
    };
    using shape_return_value_t = ttnn::Shape;
    using tensor_return_value_t = Tensor;

    struct ElementWiseMultiCore {
        static auto create(auto&&... args) {
            return element_wise_multi_core_program_factory::create(std::forward<decltype(args)>(args)...);
        }
        static void override_runtime_arguments(auto&&... args) {
            element_wise_multi_core_program_factory::override_runtime_arguments(std::forward<decltype(args)>(args)...);
        }
    };

    struct BroadcastWidthMultiCore {
        static auto create(auto&&... args) {
            return broadcast_width_multi_core_program_factory::create(std::forward<decltype(args)>(args)...);
        }
        static void override_runtime_arguments(auto&&... args) {
            broadcast_width_multi_core_program_factory::override_runtime_arguments(
                std::forward<decltype(args)>(args)...);
        }
    };

    struct BroadcastHeightMultiCore {
        static auto create(auto&&... args) {
            return broadcast_height_multi_core_program_factory::create(std::forward<decltype(args)>(args)...);
        }
        static void override_runtime_arguments(auto&&... args) {
            broadcast_height_multi_core_program_factory::override_runtime_arguments(
                std::forward<decltype(args)>(args)...);
        }
    };

    struct BroadcastHeightAndWidthMultiCore {
        static auto create(auto&&... args) {
            return broadcast_height_and_width_multi_core_program_factory::create(std::forward<decltype(args)>(args)...);
        }
        static void override_runtime_arguments(auto&&... args) {
            broadcast_height_and_width_multi_core_program_factory::override_runtime_arguments(
                std::forward<decltype(args)>(args)...);
        }
    };

    using program_factory_t = std::variant<
        ElementWiseMultiCore,
        BroadcastWidthMultiCore,
        BroadcastHeightMultiCore,
        BroadcastHeightAndWidthMultiCore>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static shape_return_value_t compute_output_shapes(
        const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t&, const tensor_args_t&);

    static operation::OpPerformanceModel create_op_performance_model(
        const operation_attributes_t& attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::binary
