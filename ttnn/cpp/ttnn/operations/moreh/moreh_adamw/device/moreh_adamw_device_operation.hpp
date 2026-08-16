// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/experimental/program_descriptor_patching.hpp>

namespace ttnn::operations::moreh::moreh_adamw {

struct MorehAdamWDeviceOperation {
    struct operation_attributes_t {
        float lr = 0.001f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;
        float weight_decay = 1e-2f;
        uint32_t step = 0;
        bool amsgrad = false;
        const MemoryConfig memory_config;
        const DeviceComputeKernelConfig compute_kernel_config;

        // lr and step are excluded from the program hash (they vary every optimizer step, so
        // hashing them would recompile every call); they are re-applied on each cache hit via
        // override_runtime_arguments(). beta1/beta2/eps/weight_decay are rarely-varying
        // hyperparameters and stay in the hash.
        static constexpr auto attribute_names = std::forward_as_tuple(
            "beta1", "beta2", "eps", "weight_decay", "amsgrad", "memory_config", "compute_kernel_config");
        auto attribute_values() const {
            return std::forward_as_tuple(
                beta1, beta2, eps, weight_decay, amsgrad, memory_config, compute_kernel_config);
        }
    };

    struct tensor_args_t {
        const Tensor& param_in;
        const Tensor& grad;
        const Tensor& exp_avg_in;
        const Tensor& exp_avg_sq_in;
        const std::optional<Tensor>& max_exp_avg_sq_in;

        const std::optional<Tensor>& param_out;
        const std::optional<Tensor>& exp_avg_out;
        const std::optional<Tensor>& exp_avg_sq_out;
        const std::optional<Tensor>& max_exp_avg_sq_out;
    };

    using spec_return_value_t = std::vector<std::optional<tt::tt_metal::TensorSpec>>;

    using tensor_return_value_t = std::vector<std::optional<Tensor>>;

    // The program factory has to live in a program_factory_t variant rather than as methods on this
    // struct: the framework's shim for factory methods declared directly on a device operation
    // (MeshDeviceOperationAdapter::DirectDescriptorFactory) covers create_descriptor only, and
    // DeviceOperationConcept accepts a factory reachable either that way or through program_factory_t.
    struct MultiCoreProgramFactory {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        // Cache-hit re-apply of all per-dispatch state, since the hash excludes lr/step: the per-core
        // runtime args AND the tensor bindings. This concept refreshes nothing on the factory's behalf —
        // the returned ProgramRunArgs is the entire update — so every TensorParameter is named here.
        // Re-derives from create_program_artifacts; see the .cpp.
        static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    using program_factory_t = std::variant<MultiCoreProgramFactory>;

    // Mandatory methods
    static void validate_inputs(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace ttnn::operations::moreh::moreh_adamw

// Prim function exposed as ttnn::prim::moreh_adamw
namespace ttnn::prim {
ttnn::operations::moreh::moreh_adamw::MorehAdamWDeviceOperation::tensor_return_value_t moreh_adamw(
    const Tensor& param_in,
    const Tensor& grad,
    const Tensor& exp_avg_in,
    const Tensor& exp_avg_sq_in,

    std::optional<float> lr,
    std::optional<float> beta1,
    std::optional<float> beta2,
    std::optional<float> eps,
    std::optional<float> weight_decay,
    std::optional<uint32_t> step,
    std::optional<bool> amsgrad,

    const std::optional<Tensor>& max_exp_avg_sq_in,
    const std::optional<Tensor>& param_out,
    const std::optional<Tensor>& exp_avg_out,
    const std::optional<Tensor>& exp_avg_sq_out,
    const std::optional<Tensor>& max_exp_avg_sq_out,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config);
}  // namespace ttnn::prim
